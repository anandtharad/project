"""Typed decision questions and results (spec §7.4, §7.6).

A decision model (Jev, a self-hosted Laya server speaking the same
protocol, or a general LLM behind the same typed interface) answers bounded
questions about the current turn. Application code owns everything around
it: which questions are asked, validation of every answer against the
question asked, confidence thresholds, and the fallback used when no valid,
confident answer is available.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DecisionKind = Literal["choice", "score", "noul"]
DecisionProvider = Literal["off", "jev", "laya", "llm"]


class QuestionSpec(BaseModel):
    """One versioned question. The wording is sent verbatim to the model."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    kind: DecisionKind
    objective: str
    instructions: str
    # choice: label -> description; score: ordered level descriptions (index =
    # score); noul: {"true": ..., "false": ...} outcome descriptions.
    criteria: dict[str, str] | list[str]
    # Value the application uses when no valid, confident answer exists:
    # a choice label, a score level, or a noul probability (0 or 1).
    fallback: str | float
    # What the application does with the answer (documentation, spec §7.4).
    downstream: str

    @model_validator(mode="after")
    def _check_answer_space(self) -> QuestionSpec:
        if self.kind == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError(f"{self.key}: a choice needs at least two labelled options")
            if self.fallback not in self.criteria:
                raise ValueError(f"{self.key}: fallback must be one of the options")
        elif self.kind == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError(f"{self.key}: a score needs at least two ordered levels")
            if not isinstance(self.fallback, (int, float)) or not (
                0 <= self.fallback <= len(self.criteria) - 1
            ):
                raise ValueError(f"{self.key}: fallback must be a level index")
        else:
            if not isinstance(self.criteria, dict) or set(self.criteria) != {"true", "false"}:
                raise ValueError(f"{self.key}: a noul describes exactly 'true' and 'false'")
            if self.fallback not in (0, 1, 0.0, 1.0):
                raise ValueError(f"{self.key}: noul fallback must be 0 or 1")
        return self

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def options(self) -> tuple[str, ...]:
        """Choice labels (empty for other kinds)."""
        return tuple(self.criteria) if self.kind == "choice" else ()

    def wire(self) -> dict[str, Any]:
        """The question as sent on the System One wire (Jev API shape)."""
        return {"type": self.kind, "instructions": self.instructions, "criteria": self.criteria}

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            {"kind": self.kind, **self.wire(), "fallback": self.fallback},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"  # an answer came back but does not fit the question asked
    MISSING = "missing"  # no answer for this question (or no response at all)


class FallbackReason(StrEnum):
    DISABLED = "disabled"  # DECISION_PROVIDER=off
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"  # connection failure, 5xx, overload
    RATE_LIMITED = "rate_limited"  # per-minute rate limit or used-up quota
    # 4xx: authentication, permission, request validation, provider moderation
    REJECTED = "rejected"
    INVALID_RESPONSE = "invalid_response"  # the response body does not match the protocol
    INVALID_ANSWER = "invalid_answer"
    MISSING_ANSWER = "missing_answer"
    LOW_CONFIDENCE = "low_confidence"  # valid, but below the policy threshold
    # Valid, but no threshold has been chosen from evaluation data for this
    # question and calibration identity (provider, model and, for an LLM, the
    # decision prompt) yet: recorded, not acted on (shadow mode).
    NOT_CALIBRATED = "not_calibrated"


class DecisionResult(BaseModel):
    """Normalized result of one question (spec §7.6). Nothing is fabricated:
    probability and confidence are present only when the model returned them."""

    model_config = ConfigDict(frozen=True)

    decision_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question_id: str
    question_version: str
    kind: DecisionKind
    # The value the application acts on (the fallback when fallback is True).
    value: str | float
    # The model's own answer when one was valid; kept for audit even when a
    # low confidence made the application use the fallback.
    returned_value: str | float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    provider: DecisionProvider
    model_id: str | None = None
    latency_ms: float | None = None
    validation_status: ValidationStatus
    fallback: bool
    fallback_reason: FallbackReason | None = None

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@{self.question_version}"


class DecisionSet(BaseModel):
    """All decisions for one turn, keyed by question id."""

    model_config = ConfigDict(frozen=True)

    provider: DecisionProvider
    results: dict[str, DecisionResult]
    latency_ms: float | None = None
    request_id: str | None = None

    def acted_on(self, question_id: str) -> DecisionResult | None:
        """The result if it is a valid, confident model answer, else None."""
        result = self.results.get(question_id)
        return result if result is not None and not result.fallback else None

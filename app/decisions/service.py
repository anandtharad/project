"""Resolve a turn's decisions: ask, validate, threshold, fall back (spec §7.6, §7.7).

`DecisionService.decide` never raises. Every question gets a result: a
valid, confident model answer, or its documented fallback with the reason.

Thresholds are per question *and* provider, because a Jev confidence and a
Laya confidence are computed differently. A question with no threshold for
the active provider runs in shadow mode: its answer is validated and stored,
but the application uses the fallback (reason `not_calibrated`) until an
evaluation on labelled data justifies acting on it (spec §7.7, §7.8).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.decisions.backend import BackendAnswers, DecisionBackendError
from app.decisions.schemas import (
    DecisionProvider,
    DecisionResult,
    DecisionSet,
    FallbackReason,
    QuestionSpec,
    ValidationStatus,
)
from app.llm.schemas import ChatMessage, ChatRole
from app.observability import log_event

logger = logging.getLogger("app.decisions")

# "Values sum to approximately 1" (Jev API reference); Laya's softmax is exact.
_PROBABILITY_SUM_TOLERANCE = 0.02
_STATE_EARLIER_MESSAGES = 4
_STATE_MESSAGE_CHARS = 600


class DecisionBackend(Protocol):
    provider: DecisionProvider
    model: str

    async def ask(
        self, state: Mapping[str, Any], questions: list[QuestionSpec]
    ) -> BackendAnswers: ...

    async def aclose(self) -> None: ...


def build_decision_state(
    history: Sequence[ChatMessage], notes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The state sent to the decision model: the latest message first (small
    models truncate from the end), a few earlier messages, and the
    conversation notes. History is oldest-first and ends with the user."""
    if not history or history[-1].role is not ChatRole.USER:
        raise ValueError("history must end with the current user message")
    earlier = [
        {
            "speaker": "person" if m.role is ChatRole.USER else "companion",
            "text": m.content[:_STATE_MESSAGE_CHARS],
        }
        for m in history[-1 - _STATE_EARLIER_MESSAGES : -1]
    ]
    state: dict[str, Any] = {"latest_message_from_person": history[-1].content}
    if earlier:
        state["earlier_messages"] = earlier
    if notes:
        state["conversation_notes"] = dict(notes)
    return state


def _finite_unit(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_distribution(probabilities: object, keys: set[str]) -> dict[str, float] | None:
    if not isinstance(probabilities, dict) or set(probabilities) != keys:
        return None
    if not all(_finite_unit(v) for v in probabilities.values()):
        return None
    if abs(sum(probabilities.values()) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        return None
    return {str(k): float(v) for k, v in probabilities.items()}


def validate_answer(
    question: QuestionSpec, answer: Mapping[str, Any]
) -> tuple[str | float, dict[str, float] | None, float | None] | None:
    """(value, probabilities, confidence) if the answer fits the question asked, else None."""
    if answer.get("type") != question.kind:
        return None
    if question.kind == "noul":
        probability = answer.get("noul")
        return (float(probability), None, None) if _finite_unit(probability) else None  # type: ignore[arg-type]
    confidence = answer.get("confidence")
    if not _finite_unit(confidence):
        return None
    assert isinstance(confidence, (int, float))
    if question.kind == "choice":
        choice = answer.get("choice")
        distribution = _valid_distribution(answer.get("probabilities"), set(question.options))
        if not isinstance(choice, str) or choice not in question.options or distribution is None:
            return None
        # "The name of the choice with the highest probability" (API reference).
        if distribution[choice] < max(distribution.values()) - 1e-6:
            return None
        return choice, distribution, float(confidence)
    levels = len(question.criteria)
    score = answer.get("score")
    distribution = _valid_distribution(answer.get("probabilities"), {str(i) for i in range(levels)})
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= levels - 1
        or distribution is None
    ):
        return None
    return float(score), distribution, float(confidence)


def _certainty(question: QuestionSpec, value: str | float, confidence: float | None) -> float:
    """What a threshold is compared against: the returned confidence for
    choice/score; for a noul (no confidence field) the probability of the
    more likely outcome."""
    if question.kind == "noul":
        assert isinstance(value, float)
        return max(value, 1.0 - value)
    assert confidence is not None
    return confidence


class DecisionService:
    def __init__(
        self,
        *,
        provider: DecisionProvider,
        backend: DecisionBackend | None,
        questions: Sequence[QuestionSpec],
        thresholds: Mapping[str, float | None],
        timeout_seconds: float,
    ) -> None:
        if (provider == "off") != (backend is None):
            raise ValueError("a backend is required unless the provider is 'off'")
        self.provider = provider
        self._backend = backend
        self._questions = list(questions)
        # question key (id@version) -> minimum certainty for this provider
        self._thresholds = dict(thresholds)
        self._timeout = timeout_seconds

    @property
    def questions(self) -> list[QuestionSpec]:
        return list(self._questions)

    async def decide(self, state: Mapping[str, Any]) -> DecisionSet:
        if self._backend is None:
            return self._all_fallback(FallbackReason.DISABLED)
        try:
            async with asyncio.timeout(self._timeout):
                raw = await self._backend.ask(state, self._questions)
        except TimeoutError:
            return self._failed(FallbackReason.TIMEOUT, "decision request timed out")
        except DecisionBackendError as exc:
            return self._failed(exc.reason, exc.message, status_code=exc.status_code)
        results = {q.id: self._resolve(q, raw) for q in self._questions}
        decision_set = DecisionSet(
            provider=self.provider,
            results=results,
            latency_ms=round(raw.latency_ms, 2),
            request_id=raw.request_id,
        )
        log_event(
            logger,
            "decisions.resolved",
            provider=self.provider,
            model_id=raw.model_id,
            latency_ms=round(raw.latency_ms, 2),
            input_tokens=raw.input_tokens,
            acted_on=sorted(k for k, r in results.items() if not r.fallback),
            fallbacks={
                k: r.fallback_reason.value
                for k, r in results.items()
                if r.fallback_reason is not None
            },
        )
        return decision_set

    def _resolve(self, question: QuestionSpec, raw: BackendAnswers) -> DecisionResult:
        common: dict[str, Any] = {
            "question_id": question.id,
            "question_version": question.version,
            "kind": question.kind,
            "provider": self.provider,
            "model_id": raw.model_id,
            "latency_ms": round(raw.latency_ms, 2),
        }
        answer = raw.answers.get(question.id)
        if answer is None:
            return DecisionResult(
                **common,
                value=question.fallback,
                validation_status=ValidationStatus.MISSING,
                fallback=True,
                fallback_reason=FallbackReason.MISSING_ANSWER,
            )
        validated = validate_answer(question, answer)
        if validated is None:
            return DecisionResult(
                **common,
                value=question.fallback,
                validation_status=ValidationStatus.INVALID,
                fallback=True,
                fallback_reason=FallbackReason.INVALID_ANSWER,
            )
        value, probabilities, confidence = validated
        threshold = self._thresholds.get(question.key)
        if threshold is None:
            reason: FallbackReason | None = FallbackReason.NOT_CALIBRATED
        elif _certainty(question, value, confidence) < threshold:
            reason = FallbackReason.LOW_CONFIDENCE
        else:
            reason = None
        return DecisionResult(
            **common,
            value=question.fallback if reason else value,
            returned_value=value,
            probabilities=probabilities,
            confidence=confidence,
            validation_status=ValidationStatus.VALID,
            fallback=reason is not None,
            fallback_reason=reason,
        )

    def _failed(
        self, reason: FallbackReason, detail: str, *, status_code: int | None = None
    ) -> DecisionSet:
        log_event(
            logger,
            "decisions.unavailable",
            level=logging.WARNING,
            provider=self.provider,
            reason=reason.value,
            status_code=status_code,
            detail=detail,
        )
        return self._all_fallback(reason)

    def _all_fallback(self, reason: FallbackReason) -> DecisionSet:
        return DecisionSet(
            provider=self.provider,
            results={
                q.id: DecisionResult(
                    question_id=q.id,
                    question_version=q.version,
                    kind=q.kind,
                    value=q.fallback,
                    provider=self.provider,
                    validation_status=ValidationStatus.MISSING,
                    fallback=True,
                    fallback_reason=reason,
                )
                for q in self._questions
            },
        )

    async def aclose(self) -> None:
        if self._backend is not None:
            await self._backend.aclose()

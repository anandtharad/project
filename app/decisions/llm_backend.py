"""Decision backend on a general LLM (DECISION_PROVIDER=llm).

A substitute for Jev behind the same typed interface: it only returns typed
answers to the versioned questions, never text for the person (spec §7.1).
One structured request per turn asks every question:

- system: the locked `decision_llm` prompt, with each question's
  instructions and options verbatim from `app/decisions/questions.py`;
- user: the decision state as JSON inside <decision_state> tags, marked as
  untrusted data (escaped so it cannot close its block);
- reply: a strict JSON schema built from the questions. A choice question
  gets an integer weight from 0 to 100 for every option; a noul question
  gets an integer probability_yes from 0 to 100.

The weights are the model's own statements, not calibrated probabilities.
Asking for probabilities that sum to 1 does not work: no structured-output
mode can require numbers to sum to 1, and in live trials gpt-oss-20b's
stated probabilities summed to 0.90-1.11, which the service's sum check
rejects; renormalising them would be repairing an answer. So the model is
asked for independent weights, and this module converts them, by a fixed
rule, into the answer shape the service validates:

- choice: `choice` is the highest-weight option (the first in option order
  on ties), `probabilities` are the weights divided by their sum, and
  `confidence` is the top normalised weight. All-zero weights carry no
  answer, so the answer is left without a confidence and the service rejects
  it (`invalid_answer`) instead of one being invented.
- noul: `noul` is probability_yes / 100.

What makes these numbers usable is a threshold chosen for this exact model
and prompt on labelled dev data and confirmed on held-out data
(`app/decisions/thresholds.py`); until then every question runs in shadow
mode.

The model sees untrusted conversation text, so nothing it returns is
trusted: the reply must match the schema exactly (any other shape is
`invalid_response`), and the service validates every answer again. Neither
the state nor the model's output is ever logged or put in an error message.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from string import Template
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError

from app.decisions.backend import BackendAnswers, DecisionBackendError
from app.decisions.schemas import DecisionProvider, FallbackReason, QuestionSpec
from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    Effort,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    ResponseSchema,
    TraceMetadata,
)
from app.observability import trace_id_var
from app.prompting import PromptError, PromptTemplate

DECISION_PROMPT_ID = "decision_llm"
NOUL_KEY = "probability_yes"
MAX_WEIGHT = 100

# Kinds this backend asks. A score question is not asked (no V1 question is
# one), so it falls back as a missing answer.
SUPPORTED_KINDS = frozenset({"choice", "noul"})

# Fragment -> the placeholders it must use (checked when the backend is built).
_FRAGMENTS: dict[str, set[str]] = {
    "question_choice": {"id", "instructions", "options"},
    "question_noul": {"id", "instructions", "yes", "no"},
    "state": {"state"},
}

# Every provider failure maps to the closest fallback reason.
_REASONS: dict[LLMErrorCategory, FallbackReason] = {
    LLMErrorCategory.TIMEOUT: FallbackReason.TIMEOUT,
    LLMErrorCategory.CONNECTION: FallbackReason.UNAVAILABLE,
    LLMErrorCategory.OVERLOADED: FallbackReason.UNAVAILABLE,
    LLMErrorCategory.SERVER_ERROR: FallbackReason.UNAVAILABLE,
    LLMErrorCategory.UNKNOWN: FallbackReason.UNAVAILABLE,
    LLMErrorCategory.RATE_LIMITED: FallbackReason.RATE_LIMITED,
    LLMErrorCategory.QUOTA_EXHAUSTED: FallbackReason.RATE_LIMITED,
    LLMErrorCategory.RESPONSE_INVALID: FallbackReason.INVALID_RESPONSE,
    LLMErrorCategory.AUTHENTICATION: FallbackReason.REJECTED,
    LLMErrorCategory.PERMISSION: FallbackReason.REJECTED,
    LLMErrorCategory.NOT_FOUND: FallbackReason.REJECTED,
    LLMErrorCategory.INVALID_REQUEST: FallbackReason.REJECTED,
    LLMErrorCategory.CONTENT_FILTERED: FallbackReason.REJECTED,
}

# Integers only (a JSON 50.0 or "50" is not an integer weight) within 0..100.
# The range is checked here, not in the schema: Groq's strict mode does not
# document `minimum` / `maximum` (docs/providers.md).
_Weight = Annotated[int, Field(strict=True, ge=0, le=MAX_WEIGHT)]
_REPLY: TypeAdapter[dict[str, dict[str, _Weight]]] = TypeAdapter(dict[str, dict[str, _Weight]])


def reason_for(error: LLMError) -> FallbackReason:
    return _REASONS[error.category]


def _keys(question: QuestionSpec) -> list[str]:
    return list(question.options) if question.kind == "choice" else [NOUL_KEY]


def build_response_schema(questions: Sequence[QuestionSpec]) -> dict[str, Any]:
    """The strict JSON schema for one reply: every property required and no
    others allowed, using only features every strict mode documents."""
    properties: dict[str, Any] = {}
    for question in questions:
        keys = _keys(question)
        properties[question.id] = {
            "type": "object",
            "additionalProperties": False,
            "required": keys,
            "properties": {key: {"type": "integer"} for key in keys},
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [q.id for q in questions],
        "properties": properties,
    }


def _fragment(prompt: PromptTemplate, name: str, **values: str) -> str:
    return Template(prompt.fragment(name)).substitute(values)


def build_system_prompt(prompt: PromptTemplate, questions: Sequence[QuestionSpec]) -> str:
    blocks = []
    for question in questions:
        if question.kind == "choice":
            assert isinstance(question.criteria, dict)
            options = "\n".join(f"- {label}: {text}" for label, text in question.criteria.items())
            blocks.append(
                _fragment(
                    prompt,
                    "question_choice",
                    id=question.id,
                    instructions=question.instructions,
                    options=options,
                )
            )
        else:
            assert isinstance(question.criteria, dict)
            blocks.append(
                _fragment(
                    prompt,
                    "question_noul",
                    id=question.id,
                    instructions=question.instructions,
                    yes=question.criteria["true"],
                    no=question.criteria["false"],
                )
            )
    return prompt.render(questions="\n\n".join(blocks))


def build_state_message(prompt: PromptTemplate, state: Mapping[str, Any]) -> str:
    # Written as \u003c and \u003e: the same characters to a JSON reader, but
    # the text can no longer close the <decision_state> block it travels in.
    data = json.dumps(dict(state), ensure_ascii=False)
    return _fragment(prompt, "state", state=data.replace("<", "\\u003c").replace(">", "\\u003e"))


def answer_from_weights(question: QuestionSpec, values: Mapping[str, int]) -> dict[str, Any]:
    """The service's answer shape (`validate_answer`) for one question.

    Choice: normalised weights. The result is a ranking with relative
    strengths, not a calibrated probability: only a threshold chosen on dev
    data for this model and prompt makes it usable.
    """
    if question.kind == "noul":
        return {"type": "noul", "noul": values[NOUL_KEY] / MAX_WEIGHT}
    weights = [values[label] for label in question.options]
    total = sum(weights)
    if total == 0:
        # No option was given any weight: there is no answer to normalise, so
        # none is invented; without a confidence the service rejects it.
        return {"type": "choice", "choice": None, "confidence": None, "probabilities": None}
    top = max(weights)
    choice = question.options[weights.index(top)]  # the first option on ties
    return {
        "type": "choice",
        "choice": choice,
        "confidence": top / total,
        "probabilities": {
            label: w / total for label, w in zip(question.options, weights, strict=True)
        },
    }


class LLMDecisionBackend:
    """Asks the typed questions of a general LLM (see the module docstring).

    Owns its LLM client (built for the decision model, separate from the reply
    model's client) and closes it in `aclose`. The answers it returns are
    normalised integer weights stated by the model, not calibrated
    probabilities; thresholds chosen on dev data are what make them usable.
    """

    provider: DecisionProvider = "llm"

    def __init__(
        self,
        *,
        llm: LLMClient,
        llm_provider: str,
        model: str,
        prompt: PromptTemplate,
        params: GenerationParams,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> None:
        for name, expected in _FRAGMENTS.items():
            found = set(Template(prompt.fragment(name)).get_identifiers())
            if found != expected:
                raise PromptError(f"{prompt.key}: fragment {name!r} must use {sorted(expected)}")
        self.model = model
        self.llm_provider = llm_provider
        self.prompt = prompt
        self.params = params
        self.max_output_tokens = max_output_tokens
        self._llm = llm
        self._timeout = timeout_seconds
        # question keys -> (system prompt, response schema); built once per set
        self._requests: dict[tuple[str, ...], tuple[str, dict[str, Any]]] = {}

    @property
    def effort(self) -> Effort | None:
        return self.params.effort

    def _request_parts(self, questions: Sequence[QuestionSpec]) -> tuple[str, dict[str, Any]]:
        key = tuple(q.key for q in questions)
        if key not in self._requests:
            self._requests[key] = (
                build_system_prompt(self.prompt, questions),
                build_response_schema(questions),
            )
        return self._requests[key]

    def request_checksum(self, questions: Sequence[QuestionSpec]) -> str:
        """SHA-256 of everything fixed in the request (system prompt and schema),
        recorded with evaluation results next to the prompt's own checksum."""
        system, schema = self._request_parts([q for q in questions if q.kind in SUPPORTED_KINDS])
        payload = json.dumps([system, schema], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def ask(self, state: Mapping[str, Any], questions: list[QuestionSpec]) -> BackendAnswers:
        asked = [q for q in questions if q.kind in SUPPORTED_KINDS]
        system, schema = self._request_parts(asked)
        request = GenerationRequest(
            model=self.model,
            system=system,
            messages=[
                ChatMessage(role=ChatRole.USER, content=build_state_message(self.prompt, state))
            ],
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self._timeout,
            params=self.params,
            trace=TraceMetadata(
                trace_id=trace_id_var.get() or f"decision-{uuid.uuid4().hex}",
                purpose=GenerationPurpose.DECISION,
            ),
            response_schema=ResponseSchema(name="decisions", json_schema=schema),
        )
        started = time.perf_counter()
        try:
            response = await self._llm.generate(request)
        except LLMError as exc:
            # Only the category and the provider's own code (a plain identifier
            # such as json_validate_failed) are passed on, never content.
            detail = f" ({exc.provider_code})" if exc.provider_code else ""
            raise DecisionBackendError(
                reason_for(exc),
                f"decision model request failed: {exc.category.value}{detail}",
                status_code=exc.status_code,
                retry_after_seconds=exc.retry_after_seconds,
                quota_exhausted=exc.category is LLMErrorCategory.QUOTA_EXHAUSTED,
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if response.finish_reason is FinishReason.MAX_TOKENS:
            raise DecisionBackendError(
                FallbackReason.INVALID_RESPONSE, "decision model output was truncated"
            )
        if response.finish_reason is FinishReason.REFUSAL:
            raise DecisionBackendError(FallbackReason.REJECTED, "decision model declined to answer")
        try:
            reply = _REPLY.validate_json(response.content)
        except ValidationError:
            # Never include the error text: it quotes the model's output.
            raise DecisionBackendError(
                FallbackReason.INVALID_RESPONSE, "decision model output is not the expected JSON"
            ) from None
        if set(reply) != {q.id for q in asked} or any(
            set(reply[q.id]) != set(_keys(q)) for q in asked
        ):
            raise DecisionBackendError(
                FallbackReason.INVALID_RESPONSE, "decision model output does not match the schema"
            )
        return BackendAnswers(
            model_id=response.model_id,
            answers={q.id: answer_from_weights(q, reply[q.id]) for q in asked},
            latency_ms=latency_ms,
            request_id=response.provider_request_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._llm.aclose()

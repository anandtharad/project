"""LLM risk classifier for the person's latest message (spec §12.2).

Runs on the existing LLM adapter with its own client (like the LLM decision
backend), by default Groq `openai/gpt-oss-safeguard-20b` at low reasoning
effort:

- system: the locked `safety_policy` prompt (the policy, in the
  gpt-oss-safeguard format);
- user: the latest message and up to four earlier messages as JSON in a
  <conversation> block, escaped so it cannot close the block, marked as
  untrusted data, the latest message labelled `latest_message_from_person`;
- reply: JSON `{"category", "rationale"}` (strict schema where the preset
  supports it, best-effort otherwise), validated with pydantic. Any other
  shape is `invalid`. The rationale may quote the message, so it is
  discarded as soon as the reply is parsed.

`classify` never raises for a provider failure: it returns a status
(`ok`, `unavailable`, `timeout`, `rate_limited`, `invalid`; `disabled` is
used by the analyzer when no classifier is configured). A failure is never
read as "safe": the analyzer switches to degraded mode. Neither the
conversation nor the model's output is ever logged or put in a result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from string import Template
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    ResponseSchema,
    TraceMetadata,
)
from app.observability import log_event, trace_id_var
from app.prompting import PromptError, PromptRegistry, PromptTemplate
from app.safety.taxonomy import SEVERITY, SafetyCategory

if TYPE_CHECKING:
    from app.config import Settings
    from app.llm.providers.registry import ResolvedProvider

logger = logging.getLogger("app.safety")

SAFETY_POLICY_PROMPT_ID = "safety_policy"
DEFAULT_POLICY_VERSION = "1.0.0"
DEFAULT_TIMEOUT_SECONDS = 4.0
DEFAULT_MAX_OUTPUT_TOKENS = 300
MAX_EARLIER_MESSAGES = 4
# Sized for Groq's free tier (2K tokens/minute on gpt-oss-safeguard-20b):
# policy, conversation and the output cap must fit one minute's budget.
EARLIER_MESSAGE_CHARS = 300
LATEST_MESSAGE_CHARS = 2000


class ClassifierStatus(StrEnum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"
    DISABLED = "disabled"


# Every provider failure maps to a status; none of them is "ok".
_STATUS: dict[LLMErrorCategory, ClassifierStatus] = {
    LLMErrorCategory.TIMEOUT: ClassifierStatus.TIMEOUT,
    LLMErrorCategory.RATE_LIMITED: ClassifierStatus.RATE_LIMITED,
    LLMErrorCategory.QUOTA_EXHAUSTED: ClassifierStatus.RATE_LIMITED,
    LLMErrorCategory.RESPONSE_INVALID: ClassifierStatus.INVALID,
    LLMErrorCategory.CONNECTION: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.OVERLOADED: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.SERVER_ERROR: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.UNKNOWN: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.AUTHENTICATION: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.PERMISSION: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.NOT_FOUND: ClassifierStatus.UNAVAILABLE,
    LLMErrorCategory.INVALID_REQUEST: ClassifierStatus.UNAVAILABLE,
    # The provider's own moderation refused the text: no label, so degraded.
    LLMErrorCategory.CONTENT_FILTERED: ClassifierStatus.UNAVAILABLE,
}


def status_for(error: LLMError) -> ClassifierStatus:
    return _STATUS[error.category]


@dataclass(frozen=True)
class ClassifierResult:
    status: ClassifierStatus
    # Set only when status is ok.
    category: SafetyCategory | None = None
    model_id: str | None = None
    latency_ms: float | None = None
    # The provider's own error code (a plain identifier), never content.
    provider_code: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ClassifierStatus.OK


DISABLED = ClassifierResult(ClassifierStatus.DISABLED)


class Classifier(Protocol):
    """What the analyzer needs from a classifier (tests use doubles)."""

    model: str

    @property
    def policy_version(self) -> str: ...

    async def classify(
        self, message: str, history: Sequence[ChatMessage] = ()
    ) -> ClassifierResult: ...

    async def aclose(self) -> None: ...


class _Label(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: SafetyCategory
    rationale: str


def label_schema() -> dict[str, Any]:
    """The reply schema. Categories run from least to most severe, so a stub
    that answers with the first enum value (the fake provider) says none."""
    categories = sorted(SafetyCategory, key=SEVERITY.__getitem__)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "rationale"],
        "properties": {
            "category": {"type": "string", "enum": [c.value for c in categories]},
            "rationale": {"type": "string"},
        },
    }


def escape_block(data: Any) -> str:
    """JSON with < and > written as \\u003c / \\u003e: the same characters to a
    JSON reader, but the text can no longer close the block it travels in."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _clip(text: str, limit: int) -> str:
    """At most `limit` characters, keeping the start and the end of long text
    (risk language is as likely at the end of a message as at the start)."""
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    return f"{text[:head]} [...] {text[-(limit - head):]}"


def build_conversation(message: str, history: Sequence[ChatMessage] = ()) -> dict[str, Any]:
    """The data the classifier labels. `history` is the earlier messages,
    oldest first, without the latest one."""
    earlier = [
        {
            "speaker": "person" if m.role is ChatRole.USER else "companion",
            "text": _clip(m.content, EARLIER_MESSAGE_CHARS),
        }
        for m in list(history)[-MAX_EARLIER_MESSAGES:]
    ]
    data: dict[str, Any] = {}
    if earlier:
        data["earlier_messages"] = earlier
    data["latest_message_from_person"] = _clip(message, LATEST_MESSAGE_CHARS)
    return data


def build_conversation_message(
    prompt: PromptTemplate, message: str, history: Sequence[ChatMessage] = ()
) -> str:
    return Template(prompt.fragment("conversation")).substitute(
        conversation=escape_block(build_conversation(message, history))
    )


def check_fragment(prompt: PromptTemplate, name: str, expected: set[str]) -> None:
    found = set(Template(prompt.fragment(name)).get_identifiers())
    if found != expected:
        raise PromptError(f"{prompt.key}: fragment {name!r} must use {sorted(expected)}")


@dataclass(frozen=True)
class LabelCall:
    """The outcome of one structured labelling request."""

    status: ClassifierStatus
    content: Mapping[str, Any] | None
    model_id: str | None
    latency_ms: float
    provider_code: str | None = None


async def run_label_request(
    llm: LLMClient,
    request: GenerationRequest,
    model_type: type[BaseModel],
    *,
    deadline_seconds: float,
) -> LabelCall:
    """Send one structured request and validate the reply against
    `model_type`. Every failure becomes a status; no content is logged or
    kept except the validated fields."""
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    try:
        async with asyncio.timeout(deadline_seconds):
            response = await llm.generate(request)
    except TimeoutError:
        return LabelCall(ClassifierStatus.TIMEOUT, None, None, elapsed())
    except LLMError as exc:
        return LabelCall(status_for(exc), None, None, elapsed(), exc.provider_code)
    latency_ms = elapsed()
    if response.finish_reason is not FinishReason.STOP:
        # Truncated, refused or stopped for another reason: no usable label.
        return LabelCall(ClassifierStatus.INVALID, None, response.model_id, latency_ms)
    try:
        parsed = model_type.model_validate_json(response.content)
    except ValidationError:
        # Never keep the error text: it quotes the model's output.
        return LabelCall(ClassifierStatus.INVALID, None, response.model_id, latency_ms)
    return LabelCall(ClassifierStatus.OK, parsed.model_dump(), response.model_id, latency_ms)


class SafetyClassifier:
    """Labels the latest message with a `SafetyCategory` (module docstring).

    Owns its LLM client (built for the classifier model, separate from the
    reply model's) and closes it in `aclose`.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        llm_provider: str,
        model: str,
        prompt: PromptTemplate,
        params: GenerationParams,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        deadline_seconds: float | None = None,
    ) -> None:
        if prompt.id != SAFETY_POLICY_PROMPT_ID:
            raise PromptError(f"{prompt.key} is not a {SAFETY_POLICY_PROMPT_ID} prompt")
        check_fragment(prompt, "conversation", {"conversation"})
        self.model = model
        self.llm_provider = llm_provider
        self.prompt = prompt
        self.params = params
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        # The provider call has its own timeout; this bounds retries too.
        self.deadline_seconds = deadline_seconds or timeout_seconds + 1.0
        self._llm = llm
        self._system = prompt.render()
        self._schema = ResponseSchema(name="safety_label", json_schema=label_schema())

    @property
    def policy_version(self) -> str:
        return self.prompt.version

    def build_request(self, message: str, history: Sequence[ChatMessage] = ()) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            system=self._system,
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content=build_conversation_message(self.prompt, message, history),
                )
            ],
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            params=self.params,
            trace=TraceMetadata(
                trace_id=trace_id_var.get() or f"safety-{uuid.uuid4().hex}",
                purpose=GenerationPurpose.SAFETY_CLASSIFICATION,
            ),
            response_schema=self._schema,
        )

    async def classify(
        self, message: str, history: Sequence[ChatMessage] = ()
    ) -> ClassifierResult:
        """`history`: the earlier messages, oldest first, without `message`."""
        call = await run_label_request(
            self._llm,
            self.build_request(message, history),
            _Label,
            deadline_seconds=self.deadline_seconds,
        )
        if call.status is not ClassifierStatus.OK or call.content is None:
            log_event(
                logger,
                "safety.classifier_failed",
                level=logging.WARNING,
                status=call.status.value,
                provider_code=call.provider_code,
                model=self.model,
            )
            return ClassifierResult(
                call.status,
                model_id=call.model_id or self.model,
                latency_ms=call.latency_ms,
                provider_code=call.provider_code,
            )
        # Only the category is kept: the rationale may quote the message.
        return ClassifierResult(
            ClassifierStatus.OK,
            category=SafetyCategory(call.content["category"]),
            model_id=call.model_id,
            latency_ms=call.latency_ms,
        )

    async def aclose(self) -> None:
        await self._llm.aclose()


def build_safety_classifier(
    settings: Settings,
    resolved: ResolvedProvider,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 0,
    temperature: float | None = None,
) -> SafetyClassifier:
    """A classifier with a dedicated client for `resolved` (see
    `resolve_auxiliary_llm`). No retries by default: on a 3-requests-per-minute
    limit a retry rarely helps, and degraded mode is the fallback."""
    from app.llm.factory import build_llm_client

    return SafetyClassifier(
        llm=build_llm_client(settings, resolved, max_retries=max_retries),
        llm_provider=resolved.preset.name,
        model=resolved.model,
        prompt=PromptRegistry().get(SAFETY_POLICY_PROMPT_ID, policy_version),
        params=GenerationParams(effort=resolved.effort, temperature=temperature),
        max_output_tokens=resolved.max_output_tokens,
        timeout_seconds=timeout_seconds,
    )

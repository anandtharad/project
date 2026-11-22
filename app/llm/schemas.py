"""Provider-neutral generation types.

Provider SDK objects never leave the provider adapters; the rest of the
application only sees these types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    # A tool's result, sent back to the model in an agent turn (app/agent/).
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call the model asked for. `arguments` is the provider's raw
    JSON text: the tool executor validates it, never the adapter."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    arguments: str = Field(max_length=8000)


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str
    # Assistant messages in an agent turn: the tool calls the model made.
    tool_calls: tuple[ToolCall, ...] = ()
    # Tool messages: which call this result answers, and the tool's name.
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """A tool offered to the model (a function with a JSON-schema argument
    object). Only adapters with `supports_tools` accept requests with tools."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=1024)
    parameters: dict[str, Any]


class GenerationPurpose(StrEnum):
    """Why a generation is requested; each purpose has its own settings."""

    CONVERSATION = "conversation"
    STATE_EXTRACTION = "state_extraction"
    MEMORY_EXTRACTION = "memory_extraction"
    # Typed decision questions answered by an LLM (DECISION_PROVIDER=llm).
    DECISION = "decision"
    # Agent turns (AGENT_MODE=on): the strategy choice and the tool loop.
    STRATEGY_SELECTION = "strategy_selection"
    AGENT_STEP = "agent_step"
    # Safety layer: the input risk classifier and the optional semantic
    # check of a reply (app/safety/).
    SAFETY_CLASSIFICATION = "safety_classification"
    SAFETY_OUTPUT_CHECK = "safety_output_check"


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: Effort | None = None
    # Not every model accepts sampling parameters; leave unset unless the
    # configured model supports it.
    # Range accepted by most providers; Anthropic (0-1) is checked at startup.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class TraceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_id: str
    purpose: GenerationPurpose
    conversation_id: str | None = None


class ResponseSchema(BaseModel):
    """Requests a JSON reply that matches `json_schema` (structured output).

    Adapters use the provider's native structured output when the preset and
    model support it, otherwise they append the schema to the system prompt
    and ask for JSON only. Either way the reply `content` is JSON text that
    the caller must still validate: no mode guarantees semantic correctness.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # A strict-compatible subset: every object lists all properties as
    # required and sets additionalProperties to false.
    json_schema: dict[str, Any]
    # Ask for constrained decoding where the provider offers it.
    strict: bool = True


class StructuredMode(StrEnum):
    """How a structured reply was obtained."""

    NATIVE_STRICT = "native_strict"  # provider-enforced schema (constrained decoding)
    NATIVE = "native"  # provider JSON mode / best-effort schema
    PROMPTED = "prompted"  # schema in the instruction only


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    system: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    params: GenerationParams = GenerationParams()
    trace: TraceMetadata
    response_schema: ResponseSchema | None = None
    # Tools the model may call. Not combined with response_schema: Groq does
    # not support structured outputs together with tool use.
    tools: tuple[ToolSpec, ...] = ()
    # "none" asks for a final answer without further tool calls.
    tool_choice: Literal["auto", "none"] | None = None

    @model_validator(mode="after")
    def _tools_or_schema(self) -> GenerationRequest:
        if self.tools and self.response_schema is not None:
            raise ValueError("tools and response_schema cannot be combined")
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice needs tools")
        return self


class FinishReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    # The model asked for tool calls (agent turns only).
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    provider: str
    requested_model: str
    # The model that actually produced the output (may differ from the
    # requested model when a provider-side fallback served the request).
    model_id: str
    finish_reason: FinishReason
    raw_finish_reason: str | None = None
    usage: TokenUsage = TokenUsage()
    provider_request_id: str | None = None
    latency_ms: float
    fallback_used: bool = False
    # True when inline reasoning (<think> blocks) had to be stripped from the
    # reply text; recorded so leaking models/configs can be identified.
    reasoning_removed: bool = False
    # Set only when the request carried a response_schema.
    structured_mode: StructuredMode | None = None
    # Set only when the request offered tools and the model called some.
    tool_calls: tuple[ToolCall, ...] = ()


class LLMErrorCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    # A daily/billing quota is used up: retrying soon will not help.
    QUOTA_EXHAUSTED = "quota_exhausted"
    # The provider's moderation rejected the input or output.
    CONTENT_FILTERED = "content_filtered"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RESPONSE_INVALID = "response_invalid"
    UNKNOWN = "unknown"


RETRYABLE_CATEGORIES = frozenset(
    {
        LLMErrorCategory.RATE_LIMITED,
        LLMErrorCategory.OVERLOADED,
        LLMErrorCategory.SERVER_ERROR,
        LLMErrorCategory.TIMEOUT,
        LLMErrorCategory.CONNECTION,
    }
)


class LLMError(Exception):
    """A normalized provider failure. `message` is safe to log (no secrets, no content)."""

    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        provider_request_id: str | None = None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.provider_request_id = provider_request_id
        # The provider's own error code (e.g. "json_validate_failed"), when it
        # is a plain identifier; safe to log, never content.
        self.provider_code = provider_code

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES

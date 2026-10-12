"""Provider-neutral generation types.

Provider SDK objects never leave the provider adapters; the rest of the
application only sees these types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str


class GenerationPurpose(StrEnum):
    """Why a generation is requested; each purpose has its own settings."""

    CONVERSATION = "conversation"


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: Effort | None = None
    # Not every model accepts sampling parameters; leave unset unless the
    # configured model supports it.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class TraceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_id: str
    purpose: GenerationPurpose
    conversation_id: str | None = None


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    system: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    params: GenerationParams = GenerationParams()
    trace: TraceMetadata


class FinishReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
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
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.provider_request_id = provider_request_id

    @property
    def retryable(self) -> bool:
        return self.category in RETRYABLE_CATEGORIES

"""Anthropic Claude adapter (official `anthropic` Python SDK, Messages API).

Request shape follows the SDK's typed `beta.messages.create` signature:
- `output_config={"effort": ...}` controls thinking depth (current Claude
  models think adaptively; there is no thinking budget to set).
- Sampling parameters are not part of the SDK 1.x signature and current models
  (Opus 4.7 and later) reject them; a configured temperature is forwarded via
  `extra_body` only for evaluating older models that still accept it.
- `fallbacks="default"` with the `server-side-fallback-2026-07-01` beta opts
  into provider-side retry on another model when the requested model declines
  for policy reasons. The model that actually served the request is recorded.

Transient errors (connection, 408/409/429/5xx) are retried by the SDK itself
with backoff that honors `retry-after`; the orchestrator bounds the total time.
"""

from __future__ import annotations

import time
from typing import Literal

import anthropic
from anthropic import omit
from anthropic.types.beta import (
    BetaFallbackBlock,
    BetaMessage,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaTextBlock,
)
from anthropic.types.beta.beta_fallback_message_iteration_usage import (
    BetaFallbackMessageIterationUsage,
)

from app.llm.schemas import (
    ChatRole,
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    LLMErrorCategory,
    TokenUsage,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.MAX_TOKENS,
    "refusal": FinishReason.REFUSAL,
}


class AnthropicLLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_retries: int = 2,
        refusal_fallback: Literal["default", "off"] = "default",
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        if client is None:
            # api_key=None lets the SDK resolve credentials from its standard sources.
            client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self._client = client
        self._refusal_fallback = refusal_fallback

    @property
    def provider(self) -> str:
        return "anthropic"

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages: list[BetaMessageParam] = [
            {"role": "user" if m.role is ChatRole.USER else "assistant", "content": m.content}
            for m in request.messages
        ]
        effort = request.params.effort
        # Current Claude models always think adaptively; "none" maps to the
        # lowest effort level they accept.
        output_config: BetaOutputConfigParam | None = (
            {"effort": "low" if effort == "none" else effort} if effort else None
        )
        use_fallback = self._refusal_fallback == "default"
        started = time.perf_counter()
        try:
            response = await self._client.beta.messages.create(
                model=request.model,
                max_tokens=request.max_output_tokens,
                system=request.system,
                messages=messages,
                output_config=output_config if output_config is not None else omit,
                fallbacks="default" if use_fallback else omit,
                betas=[FALLBACK_BETA] if use_fallback else omit,
                extra_body=(
                    {"temperature": request.params.temperature}
                    if request.params.temperature is not None
                    else None
                ),
                timeout=request.timeout_seconds,
            )
        except anthropic.AnthropicError as exc:
            raise classify_error(exc) from exc
        except TypeError as exc:
            # Raised by the SDK at request time when no credential can be resolved.
            raise LLMError(
                LLMErrorCategory.AUTHENTICATION, "Provider credentials could not be resolved."
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_response(response, requested_model=request.model, latency_ms=latency_ms)


def normalize_response(
    response: BetaMessage, *, requested_model: str, latency_ms: float
) -> GenerationResponse:
    # Thinking and other non-text blocks are never surfaced to users.
    text = "".join(block.text for block in response.content if isinstance(block, BetaTextBlock))
    fallback_used = any(isinstance(block, BetaFallbackBlock) for block in response.content) or any(
        isinstance(item, BetaFallbackMessageIterationUsage)
        for item in (response.usage.iterations or [])
    )
    raw_reason = response.stop_reason
    return GenerationResponse(
        content=text,
        provider="anthropic",
        requested_model=requested_model,
        model_id=str(response.model),
        finish_reason=_FINISH_REASONS.get(raw_reason or "", FinishReason.OTHER),
        raw_finish_reason=raw_reason,
        usage=TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
        ),
        provider_request_id=response._request_id,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def classify_error(exc: anthropic.AnthropicError) -> LLMError:
    """Map SDK exceptions to normalized categories, most specific first."""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMError(LLMErrorCategory.TIMEOUT, "Provider request timed out.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError(LLMErrorCategory.CONNECTION, "Could not connect to the provider.")
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMError(
            LLMErrorCategory.RESPONSE_INVALID,
            "Provider returned an unexpected response.",
            status_code=exc.status_code,
        )
    if isinstance(exc, anthropic.APIStatusError):
        return _classify_status_error(exc)
    return LLMError(LLMErrorCategory.UNKNOWN, "Unexpected provider error.")


def _classify_status_error(exc: anthropic.APIStatusError) -> LLMError:
    status = exc.status_code

    def error(
        category: LLMErrorCategory, message: str, *, retry_after: float | None = None
    ) -> LLMError:
        return LLMError(
            category,
            message,
            status_code=status,
            retry_after_seconds=retry_after,
            provider_request_id=exc.request_id,
        )

    if isinstance(exc, anthropic.AuthenticationError):
        return error(LLMErrorCategory.AUTHENTICATION, "Provider rejected the credentials.")
    if isinstance(exc, anthropic.PermissionDeniedError) or status == 402:
        return error(LLMErrorCategory.PERMISSION, "Provider denied the request.")
    if isinstance(exc, anthropic.NotFoundError):
        return error(LLMErrorCategory.NOT_FOUND, "Provider resource or model not found.")
    if isinstance(exc, anthropic.RateLimitError):
        return error(
            LLMErrorCategory.RATE_LIMITED,
            "Provider rate limit exceeded.",
            retry_after=_retry_after(exc),
        )
    # The SDK raises InternalServerError for 503/504, so match on status too.
    if (
        isinstance(exc, (anthropic.OverloadedError, anthropic.ServiceUnavailableError))
        or status == 503
    ):
        return error(
            LLMErrorCategory.OVERLOADED,
            "Provider is temporarily unavailable.",
            retry_after=_retry_after(exc),
        )
    if isinstance(exc, anthropic.DeadlineExceededError) or status in (408, 504):
        return error(LLMErrorCategory.TIMEOUT, "Provider request timed out.")
    if status >= 500:
        return error(LLMErrorCategory.SERVER_ERROR, "Provider server error.")
    if isinstance(
        exc,
        (
            anthropic.BadRequestError,
            anthropic.UnprocessableEntityError,
            anthropic.RequestTooLargeError,
        ),
    ):
        return error(LLMErrorCategory.INVALID_REQUEST, "Provider rejected the request.")
    return error(LLMErrorCategory.UNKNOWN, f"Provider returned HTTP {status}.")

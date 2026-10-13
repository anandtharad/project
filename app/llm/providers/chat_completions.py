"""Shared helpers for Chat Completions-style SDKs (`openai`, `groq`).

Both official SDKs are generated from the same toolkit: they expose the same
exception hierarchy (`APIStatusError` subclasses per status code, with every
5xx raised as `InternalServerError`), the same `ChatCompletion` shape, and
the same retry policy (connection errors, 408, 409, 429, 5xx, honoring
`retry-after`). The helpers here are duck-typed over that shared surface so
each adapter only adds its provider-specific request parameters.
"""

from __future__ import annotations

import re
from types import ModuleType
from typing import Any, Literal, Protocol

from app.llm.reasoning import strip_reasoning, strip_reasoning_prefix
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationResponse,
    LLMError,
    LLMErrorCategory,
    ResponseSchema,
    TokenUsage,
)
from app.llm.structured import StructuredPlan

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.MAX_TOKENS,
    "model_length": FinishReason.MAX_TOKENS,  # Mistral
    "content_filter": FinishReason.REFUSAL,
}


class ChatCompletionLike(Protocol):
    """The subset of `openai`/`groq` ChatCompletion objects we read."""

    id: str
    model: str

    @property
    def choices(self) -> list[Any]: ...

    @property
    def usage(self) -> Any: ...


def build_messages(
    system: str,
    messages: list[ChatMessage],
    *,
    system_role: Literal["system", "developer"] = "system",
) -> list[dict[str, str]]:
    """Privileged instruction first, then the conversation in order."""
    roles: dict[ChatRole, Literal["user", "assistant"]] = {
        ChatRole.USER: "user",
        ChatRole.ASSISTANT: "assistant",
    }
    return [{"role": system_role, "content": system}] + [
        {"role": roles[m.role], "content": m.content} for m in messages
    ]


def build_response_format(schema: ResponseSchema, plan: StructuredPlan) -> dict[str, Any] | None:
    """The Chat Completions `response_format` for a structured request.

    Same shape in the `openai` and `groq` SDKs (and every compatible server
    that documents structured outputs): `json_schema` carries name, schema
    and strict; `json_object` carries nothing else.
    """
    if plan.native == "schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.name,
                "schema": schema.json_schema,
                "strict": plan.strict,
            },
        }
    if plan.native == "json":
        return {"type": "json_object"}
    return None


def _content_text(content: Any) -> str:
    """Reply text from `message.content`.

    The SDKs do not validate responses, so compatible servers can return a
    list of typed parts (e.g. Mistral thinking chunks); only "text" parts are
    kept, never "thinking" parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def normalize_chat_completion(
    response: ChatCompletionLike,
    *,
    provider: str,
    requested_model: str,
    latency_ms: float,
    request_id: str | None = None,
    structured: bool = False,
) -> GenerationResponse:
    if not response.choices:
        raise LLMError(LLMErrorCategory.RESPONSE_INVALID, "Provider returned no choices.")
    choice = response.choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    raw_text = _content_text(message.content)
    # Reasoning returned in separate fields (`reasoning`, `reasoning_content`)
    # is never read; inline <think> blocks are stripped defensively.
    text, reasoning_removed = (strip_reasoning_prefix if structured else strip_reasoning)(raw_text)
    raw_reason = choice.finish_reason
    finish = _FINISH_REASONS.get(raw_reason or "", FinishReason.OTHER)
    if refusal:
        finish = FinishReason.REFUSAL
    usage = response.usage
    return GenerationResponse(
        content=text,
        provider=provider,
        requested_model=requested_model,
        model_id=response.model or requested_model,
        finish_reason=finish,
        raw_finish_reason=raw_reason,
        usage=TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cache_read_input_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
        ),
        provider_request_id=request_id or getattr(response, "_request_id", None) or response.id,
        latency_ms=latency_ms,
        reasoning_removed=reasoning_removed,
    )


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


_CODE = re.compile(r"[a-z0-9_.\-]{1,64}")


def _body_code(exc: Exception) -> str | None:
    """The error code from the response body. The openai SDK copies it to
    `exc.code`; the groq SDK leaves it in `{"error": {"code": ...}}`."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and isinstance(inner.get("code"), str):
            return str(inner["code"])
    return None


def classify_sdk_error(exc: Exception, sdk: ModuleType) -> LLMError:
    """Map an `openai`/`groq` SDK exception to a normalized error.

    `sdk` is the SDK module, whose exception classes share one design.
    Classification is by status code because the SDKs raise the same
    `InternalServerError` class for every 5xx.
    """
    if isinstance(exc, sdk.APITimeoutError):
        return LLMError(LLMErrorCategory.TIMEOUT, "Provider request timed out.")
    if isinstance(exc, sdk.APIConnectionError):
        return LLMError(LLMErrorCategory.CONNECTION, "Could not connect to the provider.")
    if isinstance(exc, sdk.APIResponseValidationError):
        return LLMError(
            LLMErrorCategory.RESPONSE_INVALID, "Provider returned an unexpected response."
        )
    if not isinstance(exc, sdk.APIStatusError):
        return LLMError(LLMErrorCategory.UNKNOWN, "Unexpected provider error.")

    status: int = exc.status_code
    headers = exc.response.headers
    request_id = headers.get("x-request-id") or headers.get("request-id")
    code = str(getattr(exc, "code", None) or _body_code(exc) or "").lower()
    text = str(exc.message).lower()
    loggable_code = code if _CODE.fullmatch(code) else None

    def error(
        category: LLMErrorCategory, message: str, *, retry_after: float | None = None
    ) -> LLMError:
        return LLMError(
            category,
            message,
            status_code=status,
            retry_after_seconds=retry_after,
            provider_request_id=request_id,
            provider_code=loggable_code,
        )

    # Some compatible providers (e.g. xAI) report an invalid key as a 400.
    if status == 401 or (status == 400 and "api key" in text):
        return error(LLMErrorCategory.AUTHENTICATION, "Provider rejected the credentials.")
    # Provider moderation reported as an error (e.g. DashScope 400
    # data_inspection_failed) rather than as a finish reason.
    if code in ("data_inspection_failed", "datainspectionfailed"):
        return error(LLMErrorCategory.CONTENT_FILTERED, "Provider moderation blocked the content.")
    if code == "arrearage":
        return error(LLMErrorCategory.PERMISSION, "Provider account is not in good standing.")
    # "insufficient_quota" means billing/free quota exhausted when the message
    # points at the plan or free allocation; DashScope also uses it for a
    # transient tokens-per-minute limit ("Allocated quota exceeded").
    if code == "insufficient_quota" and (
        "billing" in text or "free allocated quota" in text or "check your plan" in text
    ):
        return error(LLMErrorCategory.QUOTA_EXHAUSTED, "Provider quota exhausted.")
    if status in (402, 403):
        return error(LLMErrorCategory.PERMISSION, "Provider denied the request.")
    if status == 404:
        return error(LLMErrorCategory.NOT_FOUND, "Provider resource or model not found.")
    if status == 429:
        return error(
            LLMErrorCategory.RATE_LIMITED,
            "Provider rate limit exceeded.",
            retry_after=_retry_after(headers),
        )
    if status in (408, 504) or code in ("requesttimeout", "request_timeout"):
        return error(LLMErrorCategory.TIMEOUT, "Provider request timed out.")
    # 498: Groq flex-tier capacity exceeded (custom status, not retried by the SDK).
    if status in (498, 503, 529):
        return error(
            LLMErrorCategory.OVERLOADED,
            "Provider is temporarily unavailable.",
            retry_after=_retry_after(headers),
        )
    if status >= 500:
        return error(LLMErrorCategory.SERVER_ERROR, "Provider server error.")
    if status == 409:  # transient conflict; the SDKs retry it too
        return error(
            LLMErrorCategory.SERVER_ERROR,
            "Provider reported a transient conflict.",
            retry_after=_retry_after(headers),
        )
    # Groq strict structured outputs: the model's own output broke the schema
    # (400 json_validate_failed). A bad answer, not a bad request.
    if code == "json_validate_failed":
        return error(
            LLMErrorCategory.RESPONSE_INVALID, "The model's output did not match the schema."
        )
    if status in (400, 413, 422):
        return error(LLMErrorCategory.INVALID_REQUEST, "Provider rejected the request.")
    return error(LLMErrorCategory.UNKNOWN, f"Provider returned HTTP {status}.")

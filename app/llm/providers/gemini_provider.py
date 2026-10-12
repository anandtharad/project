"""Google Gemini adapter (official `google-genai` SDK, Gemini Developer API).

Verified against google-genai 2.29.0 source and an offline mock run:
- `client.aio.models.generate_content(model, contents, config)`; assistant
  turns use role "model"; the instruction goes in `system_instruction`.
- The SDK has NO default timeout and NO default retries: a timeout (integer
  milliseconds) and retry options are always set per call here.
- Thinking: Gemini 3.x takes `thinking_level` (minimal/low/medium/high; not
  every model accepts "minimal"), Gemini 2.5 takes `thinking_budget` (0
  disables on Flash models). Never both. Thought parts are excluded from the
  reply; `include_thoughts` is never enabled.
- Errors: `errors.APIError` (`ClientError` 4xx / `ServerError` 5xx) with
  `.code`, `.message`, `.details` (full body) and `.response`. An invalid key
  is reported as 400 "API key not valid". A 429 may be a per-minute quota
  (retry delay in `google.rpc.RetryInfo`) or a daily quota (do not retry).
  Transport errors arrive as raw httpx exceptions; a non-JSON 200 body as
  `json.JSONDecodeError`.
- `vertexai=False` is passed explicitly so GOOGLE_GENAI_USE_VERTEXAI in the
  environment cannot switch the client to Vertex AI.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors, types

from app.llm.reasoning import strip_reasoning
from app.llm.schemas import (
    ChatRole,
    Effort,
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    LLMErrorCategory,
    TokenUsage,
)

_REFUSAL_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}
_LENGTH_REASONS = {"MAX_TOKENS", "CONTINUATION"}

# Gemini 3 thinking levels. "minimal" is only accepted by some models (the
# flash-lite line); elsewhere "none" maps to the lowest level, "low".
_LEVELS: dict[Effort, str] = {
    "none": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# Gemini 2.5 thinking budgets (tokens). 0 disables thinking on Flash models.
_BUDGETS: dict[Effort, int] = {
    "none": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 24576,
}


def model_generation(model: str) -> Literal["2.5", "3", "other"]:
    name = model.lower().removeprefix("models/")
    if name.startswith("gemini-2.5"):
        return "2.5"
    if name.startswith("gemini-3") or name.startswith("gemini-flash"):
        return "3"
    return "other"


def thinking_config(model: str, effort: Effort | None) -> types.ThinkingConfig | None:
    if effort is None:
        return None
    generation = model_generation(model)
    if generation == "3":
        level = _LEVELS[effort]
        if effort == "none" and "flash-lite" in model.lower():
            level = "minimal"
        return types.ThinkingConfig(thinking_level=level)
    if generation == "2.5":
        budget = _BUDGETS[effort]
        if budget == 0 and "pro" in model.lower():
            budget = 128  # 2.5 Pro cannot disable thinking; 128 is its minimum.
        return types.ThinkingConfig(thinking_budget=budget)
    return None  # unknown family: leave the model's default


class GeminiLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Owning the httpx client pins the transport (the SDK would switch to
        # aiohttp if installed) and keeps exception types stable.
        self._http = http_client or httpx.AsyncClient()
        self._owns_http = http_client is None
        self._client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=types.HttpOptions(base_url=base_url, httpx_async_client=self._http),
        )
        self._retry = types.HttpRetryOptions(
            attempts=max_retries + 1, initial_delay=0.5, max_delay=8.0
        )

    @property
    def provider(self) -> str:
        return "gemini"

    async def aclose(self) -> None:
        await self._client.aio.aclose()
        if self._owns_http:
            await self._http.aclose()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        contents = [
            types.Content(
                role="user" if m.role is ChatRole.USER else "model",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in request.messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            max_output_tokens=request.max_output_tokens,
            temperature=request.params.temperature,
            thinking_config=thinking_config(request.model, request.params.effort),
            http_options=types.HttpOptions(
                timeout=int(request.timeout_seconds * 1000), retry_options=self._retry
            ),
        )
        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            raise classify_gemini_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(LLMErrorCategory.TIMEOUT, "Provider request timed out.") from exc
        except httpx.TransportError as exc:
            raise LLMError(
                LLMErrorCategory.CONNECTION, "Could not connect to the provider."
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(
                LLMErrorCategory.RESPONSE_INVALID, "Provider returned an unexpected response."
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_gemini_response(
            response, requested_model=request.model, latency_ms=latency_ms
        )


def normalize_gemini_response(
    response: types.GenerateContentResponse, *, requested_model: str, latency_ms: float
) -> GenerationResponse:
    candidate = response.candidates[0] if response.candidates else None
    raw_reason: str | None
    if candidate is None:
        # No candidate: the prompt itself was blocked (or nothing came back).
        block = response.prompt_feedback.block_reason if response.prompt_feedback else None
        raw_reason = f"BLOCKED:{block.value}" if block is not None else None
        finish = FinishReason.REFUSAL if block is not None else FinishReason.OTHER
        text = ""
    else:
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        # Thought parts are hidden reasoning and are never part of the reply.
        text = "".join(p.text for p in parts if p.text and not p.thought)
        raw_reason = candidate.finish_reason.value if candidate.finish_reason else None
        if raw_reason == "STOP":
            finish = FinishReason.STOP
        elif raw_reason in _LENGTH_REASONS:
            finish = FinishReason.MAX_TOKENS
        elif raw_reason in _REFUSAL_REASONS:
            finish = FinishReason.REFUSAL
        else:
            finish = FinishReason.OTHER
    text, reasoning_removed = strip_reasoning(text)
    usage = response.usage_metadata
    output_tokens: int | None = None
    if usage is not None and (usage.candidates_token_count or usage.thoughts_token_count):
        # Thinking tokens are billed as output, matching other providers'
        # completion-token accounting.
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return GenerationResponse(
        content=text,
        provider="gemini",
        requested_model=requested_model,
        model_id=response.model_version or requested_model,
        finish_reason=finish,
        raw_finish_reason=raw_reason,
        usage=TokenUsage(
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=output_tokens,
            cache_read_input_tokens=usage.cached_content_token_count if usage else None,
        ),
        provider_request_id=response.response_id,
        latency_ms=latency_ms,
        reasoning_removed=reasoning_removed,
    )


def _error_details(exc: errors.APIError) -> list[dict[str, Any]]:
    body = exc.details if isinstance(exc.details, dict) else {}
    error = body.get("error")
    details = error.get("details") if isinstance(error, dict) else None
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


def _retry_delay(exc: errors.APIError) -> float | None:
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    if header is not None:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    for detail in _error_details(exc):
        if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            delay = str(detail.get("retryDelay", ""))
            try:
                return max(0.0, float(delay.removesuffix("s")))
            except ValueError:
                return None
    return None


def _is_daily_quota(exc: errors.APIError) -> bool:
    for detail in _error_details(exc):
        if str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            for violation in detail.get("violations") or []:
                if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                    return True
    return False


def classify_gemini_error(exc: errors.APIError) -> LLMError:
    status = exc.code

    def error(
        category: LLMErrorCategory, message: str, *, retry_after: float | None = None
    ) -> LLMError:
        return LLMError(category, message, status_code=status, retry_after_seconds=retry_after)

    message = (exc.message or "").lower()
    if status == 401 or (status == 400 and "api key" in message):
        return error(LLMErrorCategory.AUTHENTICATION, "Provider rejected the credentials.")
    if status in (402, 403):
        return error(LLMErrorCategory.PERMISSION, "Provider denied the request.")
    if status == 404:
        return error(LLMErrorCategory.NOT_FOUND, "Provider resource or model not found.")
    if status == 429:
        if _is_daily_quota(exc):
            return error(LLMErrorCategory.QUOTA_EXHAUSTED, "Provider daily quota exhausted.")
        return error(
            LLMErrorCategory.RATE_LIMITED,
            "Provider rate limit exceeded.",
            retry_after=_retry_delay(exc),
        )
    if status in (408, 499, 504):
        return error(LLMErrorCategory.TIMEOUT, "Provider request timed out.")
    if status == 503:
        return error(LLMErrorCategory.OVERLOADED, "Provider is temporarily unavailable.")
    if status >= 500:
        return error(LLMErrorCategory.SERVER_ERROR, "Provider server error.")
    if status in (400, 413, 422):
        return error(LLMErrorCategory.INVALID_REQUEST, "Provider rejected the request.")
    return error(LLMErrorCategory.UNKNOWN, f"Provider returned HTTP {status}.")

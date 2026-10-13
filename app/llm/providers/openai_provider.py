"""OpenAI and OpenAI-compatible adapter (official `openai` SDK, Chat Completions).

Used for OpenAI itself and for providers whose official docs state OpenAI SDK
compatibility (xAI, Qwen/DashScope, DeepSeek, OpenRouter, Together, Ollama, or
a custom endpoint). Verified against openai 3.29.0 (httpx2 transport):

- `max_completion_tokens` for OpenAI (`max_tokens` is deprecated there);
  some compatible servers only accept `max_tokens`, so the preset picks.
- OpenAI wants the instruction under the "developer" role; compatible
  servers use "system".
- `reasoning_effort` is OpenAI's field; only sent when the preset says the
  endpoint understands it and an effort is configured.
- Optional params are omitted rather than sent as null (`temperature=None`
  would be serialized as JSON null).
- Refusals arrive as `message.refusal` with `content=None`.
- Non-standard reasoning fields from compatible servers (`reasoning`,
  `reasoning_content`) land in `message.model_extra` and are never read;
  inline <think> blocks are stripped defensively.

Thinking controls differ: OpenAI-style `reasoning_effort`, or DashScope's
`enable_thinking` (+ `thinking_budget`) sent as extra body fields.

Structured output (`response_schema`) follows the preset's documented
support (`structured_output`, see app/llm/structured.py): `response_format`
json_schema (strict where enforced), json_object plus the schema in the
instruction, or the instruction alone. OpenRouter structured requests also
set `provider.require_parameters` so they are never routed to an endpoint
that would silently ignore `response_format`.

Credential safety: with `api_key=None` the SDK falls back to OPENAI_API_KEY
and would send it to whatever base_url is configured; it forwards
OPENAI_ORG_ID / OPENAI_PROJECT_ID / OPENAI_CUSTOM_HEADERS to any endpoint;
and an `Authorization` line in OPENAI_CUSTOM_HEADERS overrides even an
explicit key. This adapter always passes an explicit key, pins the
Authorization header to it, and for non-OpenAI endpoints omits the other
OpenAI-only headers.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Literal, cast

import httpx2
import openai
from openai import omit
from openai.types.chat import ChatCompletionMessageParam, completion_create_params

from app.llm.providers.chat_completions import (
    build_messages,
    build_response_format,
    classify_sdk_error,
    normalize_chat_completion,
)
from app.llm.schemas import Effort, GenerationRequest, GenerationResponse
from app.llm.structured import (
    ChatStructuredOutput,
    chat_plan,
    finish_structured,
    system_with_schema,
)

# Placeholder for keyless local servers (e.g. Ollama). Never None: a None key
# would make the SDK pick up OPENAI_API_KEY from the environment.
KEYLESS_PLACEHOLDER = "not-needed"

_OPENAI_ONLY_HEADERS = ("OpenAI-Organization", "OpenAI-Project")


def build_default_headers(api_key: str, *, send_openai_headers: bool) -> dict[str, Any]:
    """Headers that pin authentication to `api_key` and, for non-OpenAI
    endpoints, remove OpenAI account headers taken from the environment."""
    headers: dict[str, Any] = {"Authorization": f"Bearer {api_key}"}
    if send_openai_headers:
        return headers
    names = list(_OPENAI_ONLY_HEADERS)
    # Parsed the same way the SDK parses it: newline-separated "Name: value".
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").split("\n"):
        name, sep, _ = line.partition(":")
        if sep and name.strip() and name.strip().lower() != "authorization":
            names.append(name.strip())
    headers.update({name: openai.Omit() for name in names})
    return headers


# DashScope thinking budgets (tokens) when thinking is enabled via effort.
_THINKING_BUDGETS: dict[Effort, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def effort_extra_body(
    effort: Effort | None,
    style: Literal["none", "reasoning_effort", "enable_thinking"],
    max_output_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Provider-specific thinking fields that travel in `extra_body`.

    DashScope's max_completion_tokens covers reasoning plus answer, so the
    thinking budget is kept below it to leave room for the reply.
    """
    if effort is None or style != "enable_thinking":
        return None
    if effort == "none":
        return {"enable_thinking": False}
    budget = _THINKING_BUDGETS[effort]
    if max_output_tokens:
        budget = min(budget, max(256, max_output_tokens - 1024))
    return {"enable_thinking": True, "thinking_budget": budget}


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        base_url: str | None,
        max_retries: int = 2,
        system_role: Literal["system", "developer"] = "system",
        max_tokens_param: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
        effort_style: Literal["none", "reasoning_effort", "enable_thinking"] = "none",
        structured_output: ChatStructuredOutput = "prompt",
        structured_require_parameters: bool = False,
        send_openai_headers: bool = False,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        key = api_key or KEYLESS_PLACEHOLDER
        default_headers = build_default_headers(key, send_openai_headers=send_openai_headers)
        self._client = openai.AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            max_retries=max_retries,
            # Omit sentinels remove headers at runtime; the hint only says str.
            default_headers=cast(Mapping[str, str], default_headers),
            http_client=http_client,
        )
        self._provider = provider
        self._system_role = system_role
        self._max_tokens_param = max_tokens_param
        self._effort_style = effort_style
        self._structured_output = structured_output
        self._structured_require_parameters = structured_require_parameters

    @property
    def provider(self) -> str:
        return self._provider

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        params = request.params
        use_completion_tokens = self._max_tokens_param == "max_completion_tokens"
        extra_body = effort_extra_body(params.effort, self._effort_style, request.max_output_tokens)
        schema = request.response_schema
        plan = chat_plan(self._structured_output, schema) if schema is not None else None
        system = request.system
        response_format: dict[str, Any] | None = None
        if schema is not None and plan is not None:
            system = system_with_schema(request.system, schema, plan)
            response_format = build_response_format(schema, plan)
            if self._structured_require_parameters:
                extra_body = {**(extra_body or {}), "provider": {"require_parameters": True}}
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=cast(
                    "list[ChatCompletionMessageParam]",
                    build_messages(system, request.messages, system_role=self._system_role),
                ),
                max_completion_tokens=request.max_output_tokens if use_completion_tokens else omit,
                max_tokens=omit if use_completion_tokens else request.max_output_tokens,
                reasoning_effort=(
                    params.effort
                    if self._effort_style == "reasoning_effort" and params.effort
                    else omit
                ),
                temperature=params.temperature if params.temperature is not None else omit,
                response_format=(
                    cast("completion_create_params.ResponseFormat", response_format)
                    if response_format is not None
                    else omit
                ),
                extra_body=extra_body,
                timeout=request.timeout_seconds,
            )
        except openai.OpenAIError as exc:
            raise classify_sdk_error(exc, openai) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return finish_structured(
            normalize_chat_completion(
                response,
                provider=self._provider,
                requested_model=request.model,
                latency_ms=latency_ms,
            ),
            plan,
        )

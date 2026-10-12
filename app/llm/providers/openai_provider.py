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

Credential safety: with `api_key=None` the SDK falls back to OPENAI_API_KEY
and would send it to whatever base_url is configured, and it forwards
OPENAI_ORG_ID / OPENAI_PROJECT_ID / OPENAI_CUSTOM_HEADERS to any endpoint.
This adapter always passes an explicit key and, for non-OpenAI endpoints,
omits those headers.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Literal, cast

import httpx2
import openai
from openai import omit

from app.llm.providers.chat_completions import (
    build_messages,
    classify_sdk_error,
    normalize_chat_completion,
)
from app.llm.schemas import GenerationRequest, GenerationResponse

# Placeholder for keyless local servers (e.g. Ollama). Never None: a None key
# would make the SDK pick up OPENAI_API_KEY from the environment.
KEYLESS_PLACEHOLDER = "not-needed"

_OPENAI_ONLY_HEADERS = ("OpenAI-Organization", "OpenAI-Project")


def openai_only_header_omissions() -> dict[str, openai.Omit]:
    """Header removals that keep OpenAI account headers off third-party endpoints."""
    names = list(_OPENAI_ONLY_HEADERS)
    # Parsed the same way the SDK parses it: newline-separated "Name: value".
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").split("\n"):
        name, sep, _ = line.partition(":")
        if sep and name.strip():
            names.append(name.strip())
    return {name: openai.Omit() for name in names}


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
        send_reasoning_effort: bool = False,
        send_openai_headers: bool = False,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        default_headers = {} if send_openai_headers else openai_only_header_omissions()
        self._client = openai.AsyncOpenAI(
            api_key=api_key or KEYLESS_PLACEHOLDER,
            base_url=base_url,
            max_retries=max_retries,
            # Omit sentinels remove headers at runtime; the hint only says str.
            default_headers=cast(Mapping[str, str], default_headers),
            http_client=http_client,
        )
        self._provider = provider
        self._system_role = system_role
        self._max_tokens_param = max_tokens_param
        self._send_reasoning_effort = send_reasoning_effort

    @property
    def provider(self) -> str:
        return self._provider

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        params = request.params
        use_completion_tokens = self._max_tokens_param == "max_completion_tokens"
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=build_messages(  # type: ignore[arg-type]
                    request.system, request.messages, system_role=self._system_role
                ),
                max_completion_tokens=request.max_output_tokens if use_completion_tokens else omit,
                max_tokens=omit if use_completion_tokens else request.max_output_tokens,
                reasoning_effort=(
                    params.effort if self._send_reasoning_effort and params.effort else omit
                ),
                temperature=params.temperature if params.temperature is not None else omit,
                timeout=request.timeout_seconds,
            )
        except openai.OpenAIError as exc:
            raise classify_sdk_error(exc, openai) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_chat_completion(
            response,
            provider=self._provider,
            requested_model=request.model,
            latency_ms=latency_ms,
        )

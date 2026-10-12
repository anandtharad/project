"""Groq adapter (official `groq` SDK, Chat Completions on api.groq.com).

Verified against groq 1.7.0 and Groq's docs (models, reasoning, errors):
- `max_completion_tokens` (`max_tokens` is deprecated).
- Reasoning controls differ by model family:
  * openai/gpt-oss-*: reasoning_effort in {low, medium, high} (no "none");
    reasoning text is returned in `message.reasoning` unless
    `include_reasoning=False`; `reasoning_format` is not supported.
  * qwen/*: reasoning_format "raw" (default) puts <think> blocks inside
    `content`, so "hidden" is always sent; reasoning_effort accepts "none"
    (reasoning off) and, on qwen3.8, low/medium/high.
  `include_reasoning` and `reasoning_format` are mutually exclusive.
- Groq returns a custom 498 (flex capacity exceeded) as a plain
  `APIStatusError`; it only occurs with service_tier=flex, which is never sent.
- The request ID is `x_groq.id` in the body.
- Reasoning is never read from `message.reasoning`, and inline <think> blocks
  are stripped defensively.
"""

from __future__ import annotations

import time
from typing import Literal, cast

import groq
from groq import omit
from groq.types.chat import ChatCompletionMessageParam

from app.llm.providers.chat_completions import (
    build_messages,
    classify_sdk_error,
    normalize_chat_completion,
)
from app.llm.schemas import Effort, GenerationRequest, GenerationResponse

GroqEffort = Literal["none", "default", "low", "medium", "high"]

# gpt-oss has no "none": the lowest setting it accepts is "low".
_GPT_OSS_EFFORT: dict[Effort, GroqEffort] = {
    "none": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_QWEN_EFFORT: dict[Effort, GroqEffort] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def model_family(model: str) -> Literal["gpt-oss", "qwen", "other"]:
    name = model.lower()
    if name.startswith("openai/gpt-oss"):
        return "gpt-oss"
    if name.startswith("qwen/"):
        return "qwen"
    return "other"


class GroqLLMClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        max_retries: int = 2,
        client: groq.AsyncGroq | None = None,
    ) -> None:
        if client is None:
            client = groq.AsyncGroq(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self._client = client

    @property
    def provider(self) -> str:
        return "groq"

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        family = model_family(request.model)
        effort = request.params.effort
        reasoning_effort: GroqEffort | None = None
        include_reasoning: bool | None = None
        reasoning_format: Literal["hidden"] | None = None
        if family == "gpt-oss":
            include_reasoning = False
            reasoning_effort = _GPT_OSS_EFFORT[effort] if effort else None
        elif family == "qwen":
            reasoning_format = "hidden"
            reasoning_effort = _QWEN_EFFORT[effort] if effort else None
        elif effort:
            # Unknown family: forward the closest Groq value; a model without
            # reasoning controls rejects it (surfaced as invalid_request).
            reasoning_effort = _QWEN_EFFORT[effort]

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=cast(
                    "list[ChatCompletionMessageParam]",
                    build_messages(request.system, request.messages),
                ),
                max_completion_tokens=request.max_output_tokens,
                temperature=(
                    request.params.temperature if request.params.temperature is not None else omit
                ),
                reasoning_effort=reasoning_effort if reasoning_effort is not None else omit,
                include_reasoning=include_reasoning if include_reasoning is not None else omit,
                reasoning_format=reasoning_format if reasoning_format is not None else omit,
                timeout=request.timeout_seconds,
            )
        except groq.GroqError as exc:
            raise classify_sdk_error(exc, groq) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = response.x_groq.id if response.x_groq is not None else None
        return normalize_chat_completion(
            response,
            provider=self.provider,
            requested_model=request.model,
            latency_ms=latency_ms,
            request_id=request_id,
        )

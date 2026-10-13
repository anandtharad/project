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
- Structured outputs (console.groq.com/docs/structured-outputs): `json_schema`
  with `strict: true` (constrained decoding) only on the models in
  `STRICT_SCHEMA_MODELS`; best-effort (`strict: false`) also on
  `BEST_EFFORT_SCHEMA_MODELS`; JSON Object Mode on all models, so every other
  model gets `json_object` with the schema in the instruction. Strict mode
  needs every property required and `additionalProperties: false`, and
  supports type arrays with "null", `anyOf`, `enum`, `$defs`/`$ref`.
  Streaming and tool use are not supported with it (neither is used here).
  The reasoning controls above stay as they are: JSON modes require
  `reasoning_format` parsed or hidden, and qwen always sends hidden.
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

import groq
from groq import omit
from groq.types.chat import ChatCompletionMessageParam, completion_create_params

from app.llm.providers.chat_completions import (
    build_messages,
    build_response_format,
    classify_sdk_error,
    normalize_chat_completion,
)
from app.llm.schemas import (
    Effort,
    GenerationRequest,
    GenerationResponse,
    ResponseSchema,
    StructuredMode,
)
from app.llm.structured import (
    StructuredPlan,
    chat_plan,
    finish_structured,
    system_with_schema,
)

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


# Groq's structured-outputs model tables (strict mode / best-effort mode).
STRICT_SCHEMA_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"})
BEST_EFFORT_SCHEMA_MODELS = STRICT_SCHEMA_MODELS | {"openai/gpt-oss-safeguard-20b"}


def structured_plan(model: str, schema: ResponseSchema) -> StructuredPlan:
    name = model.lower()
    if name in STRICT_SCHEMA_MODELS:
        return chat_plan("json_schema", schema)
    if name in BEST_EFFORT_SCHEMA_MODELS:
        # Listed for best-effort mode only, so strict is never requested.
        return StructuredPlan(StructuredMode.NATIVE, native="schema", in_instruction=False)
    return chat_plan("json_object", schema)


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

        schema = request.response_schema
        plan = structured_plan(request.model, schema) if schema is not None else None
        system = request.system
        response_format: dict[str, Any] | None = None
        if schema is not None and plan is not None:
            system = system_with_schema(request.system, schema, plan)
            response_format = build_response_format(schema, plan)

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=cast(
                    "list[ChatCompletionMessageParam]",
                    build_messages(system, request.messages),
                ),
                max_completion_tokens=request.max_output_tokens,
                temperature=(
                    request.params.temperature if request.params.temperature is not None else omit
                ),
                reasoning_effort=reasoning_effort if reasoning_effort is not None else omit,
                include_reasoning=include_reasoning if include_reasoning is not None else omit,
                reasoning_format=reasoning_format if reasoning_format is not None else omit,
                response_format=(
                    cast("completion_create_params.ResponseFormat", response_format)
                    if response_format is not None
                    else omit
                ),
                timeout=request.timeout_seconds,
            )
        except groq.GroqError as exc:
            raise classify_sdk_error(exc, groq) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = response.x_groq.id if response.x_groq is not None else None
        return finish_structured(
            normalize_chat_completion(
                response,
                provider=self.provider,
                requested_model=request.model,
                latency_ms=latency_ms,
                request_id=request_id,
                structured=plan is not None,
            ),
            plan,
        )

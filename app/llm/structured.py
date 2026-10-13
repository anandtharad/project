"""Structured output (JSON Schema replies) shared by the provider adapters.

When a request carries a `response_schema`, each adapter picks the strongest
mode its provider and model officially support (docs/providers.md lists the
choice per provider and the documentation behind it):

- NATIVE_STRICT: the provider enforces the schema (constrained decoding).
  Only used when `ResponseSchema.strict` is set.
- NATIVE: the provider's best-effort schema mode, or its JSON-object mode, in
  which case the schema is also written into the system instruction.
- PROMPTED: the schema is only described in the system instruction.

The reply content stays the model's JSON text. A surrounding Markdown code
fence is removed (it can never be part of valid JSON); nothing here parses or
validates the reply against the schema, which the caller does itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.llm.schemas import GenerationResponse, ResponseSchema, StructuredMode

# How an OpenAI-compatible (Chat Completions) endpoint is asked for JSON:
# - "json_schema": response_format json_schema; the endpoint enforces strict
#   schemas, so strict requests are NATIVE_STRICT.
# - "json_schema_best_effort": response_format json_schema, but enforcement
#   depends on the model or upstream endpoint, so the reply is NATIVE and the
#   schema is also given in the instruction.
# - "json_object": JSON-object mode only; the schema goes in the instruction.
# - "prompt": no native support; the schema goes in the instruction.
ChatStructuredOutput = Literal["json_schema", "json_schema_best_effort", "json_object", "prompt"]


@dataclass(frozen=True)
class StructuredPlan:
    """How one request asks for a JSON reply."""

    mode: StructuredMode
    # "schema": the provider's JSON Schema output; "json": JSON-object mode
    # only; None: nothing native (prompted).
    native: Literal["schema", "json"] | None
    # Whether the schema is also written into the system instruction.
    in_instruction: bool
    # The strict flag sent with a native schema (where the API has one).
    strict: bool = False


PROMPTED_PLAN = StructuredPlan(StructuredMode.PROMPTED, native=None, in_instruction=True)
JSON_MODE_PLAN = StructuredPlan(StructuredMode.NATIVE, native="json", in_instruction=True)


def enforced_schema_plan(schema: ResponseSchema) -> StructuredPlan:
    """Plan for an API whose JSON Schema mode is enforced when strict is set
    and best-effort otherwise (OpenAI-style `strict` flag)."""
    return StructuredPlan(
        StructuredMode.NATIVE_STRICT if schema.strict else StructuredMode.NATIVE,
        native="schema",
        in_instruction=False,
        strict=schema.strict,
    )


def chat_plan(style: ChatStructuredOutput, schema: ResponseSchema) -> StructuredPlan:
    if style == "json_schema":
        return enforced_schema_plan(schema)
    if style == "json_schema_best_effort":
        return StructuredPlan(
            StructuredMode.NATIVE, native="schema", in_instruction=True, strict=schema.strict
        )
    if style == "json_object":
        return JSON_MODE_PLAN
    return PROMPTED_PLAN


def schema_instruction(schema: ResponseSchema) -> str:
    """The fixed block that asks for JSON only. It contains the word "JSON",
    which JSON-object modes (OpenAI, DashScope, DeepSeek) require."""
    body = json.dumps(schema.json_schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "Reply with only a JSON object that conforms to the JSON Schema below. "
        "Do not add any other text and do not wrap it in a Markdown code fence.\n"
        f"JSON Schema: {body}"
    )


def system_with_schema(system: str, schema: ResponseSchema, plan: StructuredPlan) -> str:
    """The system instruction to send: unchanged unless the plan needs the schema in it."""
    if not plan.in_instruction:
        return system
    instruction = schema_instruction(schema)
    return f"{system}\n\n{instruction}" if system else instruction


_FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```\s*\Z", re.IGNORECASE | re.DOTALL)


def strip_json_fence(text: str) -> str:
    """Remove a Markdown code fence wrapping the whole reply, if present."""
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def finish_structured(
    response: GenerationResponse, plan: StructuredPlan | None
) -> GenerationResponse:
    """Record how a structured reply was obtained and unwrap a fenced reply.

    Returns `response` unchanged for requests without a response_schema.
    """
    if plan is None:
        return response
    return response.model_copy(
        update={"content": strip_json_fence(response.content), "structured_mode": plan.mode}
    )

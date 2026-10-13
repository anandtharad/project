"""DEVELOPMENT STUB - not a language model.

Returns a fixed, clearly labeled reply so the service can be run end to end
without provider credentials. It never echoes user content. Configuration
refuses this provider when APP_ENV=production.

With a response_schema it returns a deterministic minimal instance of the
schema instead (reported as NATIVE: a stub, not a real structured output).
"""

from __future__ import annotations

import json
from typing import Any

from app.llm.schemas import (
    ChatRole,
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    StructuredMode,
    TokenUsage,
)

_SCALARS: dict[str, Any] = {"string": "", "integer": 0, "number": 0, "boolean": False}


def minimal_instance(schema: Any) -> Any:
    """The smallest value of `schema`: an object holds its required properties;
    strings are "", arrays [], numbers 0, booleans false; an enum gives its
    first value and a nullable schema null. Unsupported keywords give null."""
    if not isinstance(schema, dict):
        return None
    if "const" in schema:
        return schema["const"]
    types = schema.get("type")
    options = schema.get("anyOf") or schema.get("oneOf") or []
    if (isinstance(types, list) and "null" in types) or any(
        isinstance(o, dict) and o.get("type") == "null" for o in options
    ):
        return None
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    if options:
        return minimal_instance(options[0])
    kind = types[0] if isinstance(types, list) and types else types
    if kind == "object":
        properties = schema.get("properties") or {}
        return {
            name: minimal_instance(properties.get(name)) for name in schema.get("required") or []
        }
    if kind == "array":
        return []
    return _SCALARS.get(kind) if isinstance(kind, str) else None


class FakeLLMClient:
    @property
    def provider(self) -> str:
        return "fake"

    async def aclose(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        structured_mode: StructuredMode | None = None
        if request.response_schema is not None:
            content = json.dumps(minimal_instance(request.response_schema.json_schema))
            structured_mode = StructuredMode.NATIVE
        else:
            user_turns = sum(1 for m in request.messages if m.role is ChatRole.USER)
            content = (
                "[development stub - no language model was called] "
                f"Received your message ({len(request.messages[-1].content)} characters); "
                f"{user_turns} user turn(s) are in context."
            )
        return GenerationResponse(
            content=content,
            provider=self.provider,
            requested_model=request.model,
            model_id="fake-stub",
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=TokenUsage(),
            latency_ms=0.0,
            structured_mode=structured_mode,
        )

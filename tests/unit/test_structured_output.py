"""Shared structured-output helpers and the development stub's JSON replies.

Adapter request/response shapes are tested in each test_<provider>_provider.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.llm.providers.fake import FakeLLMClient, minimal_instance
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationPurpose,
    GenerationRequest,
    ResponseSchema,
    StructuredMode,
    TraceMetadata,
)
from app.llm.structured import (
    JSON_MODE_PLAN,
    PROMPTED_PLAN,
    ChatStructuredOutput,
    chat_plan,
    schema_instruction,
    strip_json_fence,
    system_with_schema,
)
from tests.unit.structured_fixtures import STATE_JSON, STATE_SCHEMA, state_schema


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{STATE_JSON}\n```",
        f"```JSON\n{STATE_JSON}\n```\n",
        f"```\n{STATE_JSON}\n```",
        f"  ```json {STATE_JSON}```  ",
        f"```json\n\n{STATE_JSON}\n\n```",
    ],
)
def test_surrounding_fence_is_removed(text: str) -> None:
    assert strip_json_fence(text) == STATE_JSON


@pytest.mark.parametrize(
    "text",
    [
        STATE_JSON,
        f" {STATE_JSON}\n",
        f"Here you go:\n```json\n{STATE_JSON}\n```",  # prose outside: not a pure fence
        '{"note": "```json inside a string```"}',
        "",
    ],
)
def test_other_text_is_left_unchanged(text: str) -> None:
    assert strip_json_fence(text) == text


def test_schema_instruction_is_fixed_and_compact() -> None:
    instruction = schema_instruction(state_schema())
    assert instruction.startswith("Reply with only a JSON object")
    assert instruction.endswith("JSON Schema: " + json.dumps(STATE_SCHEMA, separators=(",", ":")))
    assert "\n\n" not in instruction


def test_system_with_schema() -> None:
    schema = state_schema()
    instruction = schema_instruction(schema)
    assert system_with_schema("SYSTEM", schema, PROMPTED_PLAN) == f"SYSTEM\n\n{instruction}"
    assert system_with_schema("", schema, JSON_MODE_PLAN) == instruction
    strict = chat_plan("json_schema", schema)
    assert system_with_schema("SYSTEM", schema, strict) == "SYSTEM"


@pytest.mark.parametrize(
    ("style", "strict", "expected"),
    [
        ("json_schema", True, (StructuredMode.NATIVE_STRICT, "schema", False, True)),
        ("json_schema", False, (StructuredMode.NATIVE, "schema", False, False)),
        ("json_schema_best_effort", True, (StructuredMode.NATIVE, "schema", True, True)),
        ("json_schema_best_effort", False, (StructuredMode.NATIVE, "schema", True, False)),
        ("json_object", True, (StructuredMode.NATIVE, "json", True, False)),
        ("prompt", True, (StructuredMode.PROMPTED, None, True, False)),
    ],
)
def test_chat_plan(
    style: ChatStructuredOutput,
    strict: bool,
    expected: tuple[StructuredMode, str | None, bool, bool],
) -> None:
    plan = chat_plan(style, state_schema(strict=strict))
    assert (plan.mode, plan.native, plan.in_instruction, plan.strict) == expected


def test_minimal_instance_of_the_state_schema() -> None:
    assert minimal_instance(STATE_SCHEMA) == {"topic": None, "subtopics": [], "user_goal": None}


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string"}, ""),
        ({"type": "integer"}, 0),
        ({"type": "number"}, 0),
        ({"type": "boolean"}, False),
        ({"type": "array", "items": {"type": "integer"}}, []),
        ({"type": "null"}, None),
        ({"type": "string", "enum": ["stated", "inferred", "unknown"]}, "stated"),
        ({"type": ["integer", "null"]}, None),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, None),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, 0),
        ({"const": "v1"}, "v1"),
        ({"type": ["string", "integer"]}, ""),
        ({"type": "object", "properties": {"a": {"type": "string"}}}, {}),
        (
            {
                "type": "object",
                "required": ["inner"],
                "properties": {
                    "inner": {
                        "type": "object",
                        "required": ["n", "flag"],
                        "properties": {"n": {"type": "integer"}, "flag": {"type": "boolean"}},
                    },
                    "optional": {"type": "string"},
                },
            },
            {"inner": {"n": 0, "flag": False}},
        ),
        ({"$ref": "#/$defs/x"}, None),  # unsupported keyword
    ],
)
def test_minimal_instance(schema: dict[str, Any], expected: Any) -> None:
    assert minimal_instance(schema) == expected


def stub_request(schema: ResponseSchema | None) -> GenerationRequest:
    return GenerationRequest(
        model="fake-stub",
        system="SYSTEM",
        messages=[ChatMessage(role=ChatRole.USER, content="hello")],
        max_output_tokens=100,
        timeout_seconds=5,
        trace=TraceMetadata(trace_id="t", purpose=GenerationPurpose.STATE_EXTRACTION),
        response_schema=schema,
    )


async def test_fake_provider_returns_minimal_json_for_a_schema() -> None:
    response = await FakeLLMClient().generate(stub_request(state_schema()))
    assert json.loads(response.content) == {"topic": None, "subtopics": [], "user_goal": None}
    assert response.structured_mode is StructuredMode.NATIVE


async def test_fake_provider_without_schema_is_unchanged() -> None:
    response = await FakeLLMClient().generate(stub_request(None))
    assert response.content.startswith("[development stub - no language model was called]")
    assert response.structured_mode is None

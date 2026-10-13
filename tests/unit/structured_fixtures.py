"""Shared data for the structured-output tests: the JSON Schema the state
extractor sends, and a reply that matches it."""

from __future__ import annotations

import json
from typing import Any

from app.llm.schemas import ResponseSchema

STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topic", "subtopics", "user_goal"],
    "properties": {
        "topic": {"type": ["string", "null"]},
        "subtopics": {"type": "array", "items": {"type": "string"}},
        "user_goal": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "provenance"],
                    "properties": {
                        "value": {"type": "string"},
                        "provenance": {"type": "string", "enum": ["stated", "inferred", "unknown"]},
                    },
                },
            ]
        },
    },
}

STATE_JSON = json.dumps(
    {
        "topic": "work",
        "subtopics": ["manager"],
        "user_goal": {"value": "talk to the manager", "provenance": "stated"},
    }
)


def state_schema(*, strict: bool = True) -> ResponseSchema:
    return ResponseSchema(name="conversation_state", json_schema=STATE_SCHEMA, strict=strict)


def fenced(text: str) -> str:
    """`text` wrapped the way models often wrap JSON when only prompted."""
    return f"```json\n{text}\n```"

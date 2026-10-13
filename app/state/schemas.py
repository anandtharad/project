"""Conversation state (spec §8.1, §8.2).

State is a compact, typed summary of the conversation that the application
keeps between turns. Every field may be absent. Goals and emotional context
carry provenance, so an inference is never stored as if the person had said
it. State is derived from untrusted conversation text: it is context for the
model, never instructions, and never a source of verified personal facts.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# "memory" (retrieved from user memory) arrives with Phase 3.
Provenance = Literal["stated", "inferred", "unknown"]

STATE_SCHEMA_VERSION = 1
MAX_TEXT = 200
MAX_ITEMS = 5
MAX_ITEM_TEXT = 120


def _clean(text: str, limit: int) -> str | None:
    # Drop control characters: PostgreSQL rejects NUL in jsonb, which would
    # fail the whole completion transaction.
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\t\n")
    text = " ".join(text.split())
    return text[:limit] or None


class InferredField(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str = Field(min_length=1, max_length=MAX_TEXT)
    provenance: Literal["stated", "inferred"]


class ConversationState(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str | None = Field(default=None, max_length=MAX_TEXT)
    subtopics: tuple[str, ...] = Field(default=(), max_length=MAX_ITEMS)
    user_goal: InferredField | None = None
    # Set from a validated decision (conversation_mode), not by extraction.
    conversation_mode: str | None = None
    emotional_context: InferredField | None = None
    open_questions: tuple[str, ...] = Field(default=(), max_length=MAX_ITEMS)
    recently_discussed: tuple[str, ...] = Field(default=(), max_length=MAX_ITEMS)

    def notes(self) -> dict[str, Any]:
        """The non-empty fields, as shown to the models (untrusted context)."""
        data = self.model_dump(mode="json", exclude_none=True)
        return {k: v for k, v in data.items() if v not in ((), [], "")}

    def is_empty(self) -> bool:
        return not self.notes()


class StoredState(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: ConversationState
    version: int = Field(ge=1)
    updated_at: datetime | None = None


# --- What the extraction model returns ---------------------------------------
#
# The wire schema avoids nulls and anyOf so every structured-output mode can
# enforce it: an empty string means "none", and provenance "unknown" means
# the field is not known. `to_state` normalizes it into ConversationState.


class ExtractedField(BaseModel):
    value: str
    provenance: Provenance

    def to_field(self) -> InferredField | None:
        value = _clean(self.value, MAX_TEXT)
        if value is None or self.provenance == "unknown":
            return None
        return InferredField(value=value, provenance=self.provenance)


class ExtractedState(BaseModel):
    topic: str
    subtopics: list[str]
    user_goal: ExtractedField
    emotional_context: ExtractedField
    open_questions: list[str]
    recently_discussed: list[str]

    @field_validator("subtopics", "open_questions", "recently_discussed")
    @classmethod
    def _items(cls, items: list[str]) -> list[str]:
        cleaned = [c for c in (_clean(i, MAX_ITEM_TEXT) for i in items) if c]
        return list(dict.fromkeys(cleaned))[:MAX_ITEMS]

    def to_state(self, conversation_mode: str | None) -> ConversationState:
        return ConversationState(
            topic=_clean(self.topic, MAX_TEXT),
            subtopics=tuple(self.subtopics),
            user_goal=self.user_goal.to_field(),
            conversation_mode=conversation_mode,
            emotional_context=self.emotional_context.to_field(),
            open_questions=tuple(self.open_questions),
            recently_discussed=tuple(self.recently_discussed),
        )


_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value", "provenance"],
    "properties": {
        "value": {"type": "string"},
        "provenance": {"type": "string", "enum": ["stated", "inferred", "unknown"]},
    },
}
_LIST_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "topic",
        "subtopics",
        "user_goal",
        "emotional_context",
        "open_questions",
        "recently_discussed",
    ],
    "properties": {
        "topic": {"type": "string"},
        "subtopics": _LIST_SCHEMA,
        "user_goal": _FIELD_SCHEMA,
        "emotional_context": _FIELD_SCHEMA,
        "open_questions": _LIST_SCHEMA,
        "recently_discussed": _LIST_SCHEMA,
    },
}

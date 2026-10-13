"""Memory types and the extraction model's output (spec §10.2-10.4).

A memory is something a person explicitly shared about themselves that is
useful in future conversations. The extraction model only proposes
candidates; `app/memory/policy.py` decides what is stored, updated, held
for confirmation, or rejected.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MEMORY_SCHEMA_VERSION = 1


class MemoryType(StrEnum):
    EPISODIC = "episodic"  # an ongoing or upcoming situation, a recent event
    SEMANTIC = "semantic"  # a stable fact the person shared
    PREFERENCE = "preference"  # how they want the companion to talk with them


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    # Conflicts with an existing memory and the new information does not
    # clearly supersede it: kept out of retrieval until the person confirms.
    PENDING_CONFIRMATION = "pending_confirmation"
    SUPERSEDED = "superseded"  # replaced by a newer memory (provenance kept)


SensitiveCategory = Literal[
    "none",
    "health",
    "mental_health",
    "sexuality",
    "religion",
    "politics",
    "ethnicity",
    "legal",
    "finance",
    "identifier",
]
Lifetime = Literal["short", "medium", "long"]
Relation = Literal["new", "duplicate", "updates", "conflicts"]


class Memory(BaseModel):
    """A stored memory as the application sees it."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    memory_type: MemoryType
    content: str
    status: MemoryStatus
    importance: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    source_conversation_id: uuid.UUID | None = None
    supersedes_id: uuid.UUID | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory: Memory
    score: float
    semantic: float | None = None
    lexical: float | None = None


# --- What the extraction model returns ---------------------------------------
#
# Strict-schema friendly: every field required, no nulls, no numeric bounds
# (validated in code). `related_memory` indexes the existing memories shown
# to the model, or is -1.


class ExtractedCandidate(BaseModel):
    content: str
    memory_type: MemoryType
    # The person said it themselves in the latest message (not an inference).
    explicit: bool
    # A short quote from the latest message that supports the memory.
    evidence: str
    sensitive_category: SensitiveCategory
    lifetime: Lifetime
    importance: int
    relation: Relation
    related_memory: int


class ExtractionOutput(BaseModel):
    candidates: list[ExtractedCandidate]


MEMORY_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "memory_type",
                    "explicit",
                    "evidence",
                    "sensitive_category",
                    "lifetime",
                    "importance",
                    "relation",
                    "related_memory",
                ],
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string", "enum": [t.value for t in MemoryType]},
                    "explicit": {"type": "boolean"},
                    "evidence": {"type": "string"},
                    "sensitive_category": {
                        "type": "string",
                        "enum": list(SensitiveCategory.__args__),  # type: ignore[attr-defined]
                    },
                    "lifetime": {"type": "string", "enum": ["short", "medium", "long"]},
                    "importance": {"type": "integer"},
                    "relation": {
                        "type": "string",
                        "enum": ["new", "duplicate", "updates", "conflicts"],
                    },
                    "related_memory": {"type": "integer"},
                },
            },
        }
    },
}

"""The canonical training record (spec §17.4), independent of any tokenizer
or chat template.

One record is one target reply: the system instruction (by reference), the
history before it, the person's message, optional state, strategy, memory
and knowledge context, the target, provenance, and quality and safety
labels. An optional field that does not apply is None **and** listed in
`absent`, so a reader can tell "not applicable" from "lost"; nothing is ever
made up to fill a field (no invented memories or evidence).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
Role = Literal["user", "assistant"]
Split = Literal["train", "validation"]
OPTIONAL_FIELDS = ("state", "strategy", "memory", "knowledge")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(min_length=1)


class SystemRef(BaseModel):
    """The system instruction, by reference: the exporter renders it from the
    prompt registry, so records do not carry a copy that could drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str  # "<id>@<version>"
    checksum: str


class Strategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    # Where the label came from, and the source's own labels.
    annotation: Literal["source_label"]
    source_labels: list[str]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str  # training/sources.toml id
    source_url: str
    licence: str
    file_sha256: str
    conversation: str  # the source's own conversation id
    turn: int  # index of the target among the joined messages
    synthetic: bool
    # Every transformation applied, in order, with its version.
    transformations: list[str]
    pipeline_version: str


class QualityLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_words: int
    message_words: int
    history_messages: int
    history_truncated: bool
    # Labels the source itself provides (survey scores, emotion, ...).
    source: dict[str, Any] = Field(default_factory=dict)


class SafetyLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Rules of the deterministic screen (app/safety/patterns.py) that fired
    # on the message or the target; a record is only kept when none did.
    screen_version: str
    message_hits: list[str]
    target_hits: list[str]


class TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    id: str
    split: Split
    system: SystemRef
    history: list[Message]
    message: str = Field(min_length=1)
    target: str = Field(min_length=1)
    state: dict[str, Any] | None = None
    strategy: Strategy | None = None
    memory: list[str] | None = None
    knowledge: list[dict[str, str]] | None = None
    absent: list[str]
    provenance: Provenance
    quality: QualityLabels
    safety: SafetyLabels

    @model_validator(mode="after")
    def _absent_is_exact(self) -> TrainingRecord:
        missing = sorted(f for f in OPTIONAL_FIELDS if getattr(self, f) is None)
        if sorted(self.absent) != missing:
            raise ValueError(f"absent must list exactly the empty optional fields: {missing}")
        if self.history and self.history[0].role != "user":
            raise ValueError("history starts with the person")
        return self

"""Public request/response models for /v1."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Hard schema ceiling; the configurable MAX_INPUT_CHARACTERS (lower) is
# enforced in the route so it can change without a code change.
_HARD_MAX_CONTENT = 100_000


class ConversationCreated(BaseModel):
    conversation_id: uuid.UUID
    created_at: datetime
    status: str


class ConversationDetail(BaseModel):
    conversation_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    status: str
    message_count: int


class SubmitMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(max_length=_HARD_MAX_CONTENT)
    idempotency_key: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )

    @field_validator("content")
    @classmethod
    def _content_is_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        if "\x00" in value:
            raise ValueError("content must not contain NUL characters")
        return value


class SupportContact(BaseModel):
    name: str
    contact: str
    hours: str


class SafetyNotice(BaseModel):
    pathway: str = Field(
        description="The safety pathway the reply followed (docs/safety.md), e.g. crisis_support."
    )
    resources: list[SupportContact] = Field(
        description="Verified support contacts for the configured region (SAFETY_REGION); "
        "a client may show them alongside the reply. Empty for gentle pathways."
    )


class CitedSource(BaseModel):
    marker: str = Field(description="The marker used in the reply text, e.g. S1 for [S1].")
    title: str
    publisher: str
    url: str
    section: str = Field(description="Heading path of the cited passage.")
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    version: int = Field(description="Version of the source document that was cited.")


class SubmitMessageResponse(BaseModel):
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    assistant_content: str
    created_at: datetime
    replayed: bool = Field(
        description="True when this response replays a previously completed request "
        "with the same idempotency key."
    )
    timing_ms: dict[str, float] | None = Field(
        default=None, description="Development only: elapsed time at each turn stage."
    )
    safety: SafetyNotice | None = Field(
        default=None,
        description="Present only when the reply followed a safety pathway (not the ordinary one).",
    )
    sources: list[CitedSource] = Field(
        default_factory=list,
        description="Reference passages cited in the reply ([S1] markers), from the reviewed "
        "knowledge library. Empty when the reply cites nothing.",
    )


class MessageOut(BaseModel):
    message_id: uuid.UUID
    sequence: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    status: Literal["pending", "completed", "failed"] | None = Field(
        default=None, description="Turn status; present on user messages only."
    )
    sources: list[CitedSource] = Field(
        default_factory=list, description="Sources cited in an assistant reply."
    )


class MessageList(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageOut]
    next_after_sequence: int | None = Field(
        default=None, description="Pass as after_sequence to fetch the next page."
    )

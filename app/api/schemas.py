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


class MessageOut(BaseModel):
    message_id: uuid.UUID
    sequence: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    status: Literal["pending", "completed", "failed"] | None = Field(
        default=None, description="Turn status; present on user messages only."
    )


class MessageList(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageOut]
    next_after_sequence: int | None = Field(
        default=None, description="Pass as after_sequence to fetch the next page."
    )

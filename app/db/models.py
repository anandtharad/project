"""Relational schema (SQLAlchemy 2.x declarative models).

Phase 1: users, conversations, messages. Phase 2: conversation_states (one
versioned state per conversation) and turn_decisions (one row per decision
question per completed turn). Later phases add their own tables through new
migrations.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # Fetch server-generated defaults (timestamps) via RETURNING on insert, so
    # they never trigger implicit lazy loads under asyncio.
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TurnStatus(StrEnum):
    """Lifecycle of a user message's turn. Only user messages carry a status."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Subject identifier from the authentication provider.
    external_id: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), server_default=text("'active'"))
    # Highest message sequence allocated so far. Updated only while the
    # conversation row is locked, so sequences are gap-free and ordered.
    last_sequence: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_messages_conversation_idempotency_key"
        ),
        CheckConstraint("role IN ('user', 'assistant')", name="role_valid"),
        CheckConstraint("sequence > 0", name="sequence_positive"),
        CheckConstraint(
            "(role = 'user' AND turn_status IN ('pending', 'completed', 'failed')"
            " AND reply_to_message_id IS NULL)"
            " OR (role = 'assistant' AND turn_status IS NULL AND idempotency_key IS NULL"
            " AND reply_to_message_id IS NOT NULL AND model_id IS NOT NULL"
            " AND prompt_id IS NOT NULL AND prompt_version IS NOT NULL)",
            name="role_fields",
        ),
        # At most one in-flight turn per conversation, enforced by the database.
        Index(
            "uq_messages_one_pending_turn",
            "conversation_id",
            unique=True,
            postgresql_where=text("turn_status = 'pending'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # --- User-message turn tracking --------------------------------------
    turn_status: Mapped[str | None] = mapped_column(String(16))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    # SHA-256 of the content: detects an idempotency key reused for a different message.
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    # Identifies the generation attempt allowed to complete this turn.
    attempt_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    attempt_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[str | None] = mapped_column(String(64))

    # --- Assistant-message provenance ------------------------------------
    # Unique: a user message can have at most one assistant reply.
    reply_to_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), unique=True
    )
    provider: Mapped[str | None] = mapped_column(String(64))
    model_id: Mapped[str | None] = mapped_column(String(128))
    prompt_id: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    generation_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ConversationStateRecord(Base):
    """The latest conversation state. `state_version` increases by one on
    every accepted write; a write based on an older version is rejected."""

    __tablename__ = "conversation_states"
    __table_args__ = (CheckConstraint("state_version > 0", name="state_version_positive"),)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    schema_version: Mapped[int] = mapped_column(Integer)
    state: Mapped[dict[str, Any]] = mapped_column(JSONB)
    state_version: Mapped[int] = mapped_column(Integer)
    # The user message whose completed turn produced this state.
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    # How it was produced: extraction model, prompt, structured-output mode.
    extraction: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TurnDecision(Base):
    """One decision result (spec §7.6), stored with the completed turn."""

    __tablename__ = "turn_decisions"
    __table_args__ = (
        UniqueConstraint(
            "user_message_id", "question_id", name="uq_turn_decisions_message_question"
        ),
        CheckConstraint("kind IN ('choice', 'score', 'noul')", name="kind_valid"),
        CheckConstraint(
            "validation_status IN ('valid', 'invalid', 'missing')", name="validation_status_valid"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(16))
    # A choice label or a number (score level / noul probability).
    value: Mapped[Any] = mapped_column(JSONB)
    returned_value: Mapped[Any | None] = mapped_column(JSONB)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(16))
    model_id: Mapped[str | None] = mapped_column(String(128))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    validation_status: Mapped[str] = mapped_column(String(16))
    fallback: Mapped[bool] = mapped_column(Boolean)
    fallback_reason: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

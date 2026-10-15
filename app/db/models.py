"""Relational schema (SQLAlchemy 2.x declarative models).

Phase 1: users, conversations, messages. Phase 2: conversation_states (one
versioned state per conversation) and turn_decisions (one row per decision
question per completed turn). Phase 3: memories (per user, with an optional
embedding) and user_memory_settings. Later phases add their own tables
through new migrations.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
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


class TurnSafety(Base):
    """The safety layer's record for one completed turn (spec §12, design in
    docs/safety.md): labels, statuses and rule ids only, never message text
    or classifier rationale. Deleted with its conversation."""

    __tablename__ = "turn_safety"
    __table_args__ = (
        CheckConstraint("recovery IN ('none', 'regenerated', 'fallback')", name="recovery_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    user_message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), unique=True
    )
    category: Mapped[str] = mapped_column(String(32))
    pathway: Mapped[str] = mapped_column(String(32))
    # False: the classifier failed and no pattern matched ("unassessed").
    assessed: Mapped[bool] = mapped_column(Boolean)
    degraded: Mapped[bool] = mapped_column(Boolean)
    classifier_status: Mapped[str] = mapped_column(String(16))
    classifier_model: Mapped[str | None] = mapped_column(String(128))
    classifier_category: Mapped[str | None] = mapped_column(String(32))
    pattern_rules: Mapped[list[str]] = mapped_column(JSONB)
    decision_signal: Mapped[bool] = mapped_column(Boolean)
    policy_version: Mapped[str] = mapped_column(String(32))
    patterns_version: Mapped[str] = mapped_column(String(32))
    region: Mapped[str] = mapped_column(String(8))
    # Violation codes of the stored reply's first draft (empty: none).
    output_violations: Mapped[list[str]] = mapped_column(JSONB)
    output_check_status: Mapped[str | None] = mapped_column(String(16))
    recovery: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryRecord(Base):
    """A memory about a user (spec §10.4). Owned by the user, not by a
    conversation: deleting a conversation keeps its memories (the person
    manages them separately), deleting the user deletes them. The embedding
    lives in the same row, so deleting a memory deletes its vector too."""

    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint(
            "memory_type IN ('episodic', 'semantic', 'preference')", name="memory_type_valid"
        ),
        CheckConstraint(
            "status IN ('active', 'pending_confirmation', 'superseded')", name="status_valid"
        ),
        CheckConstraint("importance BETWEEN 1 AND 3", name="importance_range"),
        CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)", name="embedding_has_model"
        ),
        Index("ix_memories_user_status", "user_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    memory_type: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24))
    importance: Mapped[int] = mapped_column(SmallInteger)
    # explicit flag, evidence quote, extraction model and prompt, policy reason.
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    # The memory this one replaced (status superseded), kept for provenance.
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Vectors from different models are never compared; dimensions vary by model.
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    schema_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserMemorySettings(Base):
    """Per-user memory preferences (spec §10.7). Absent row = defaults."""

    __tablename__ = "user_memory_settings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    memory_enabled: Mapped[bool] = mapped_column(Boolean, server_default=text("true"))
    allow_sensitive: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    # Last "delete all": memories from messages sent before it are not stored.
    memories_cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeDocument(Base):
    """One version of a reviewed knowledge source (spec §11.3). A changed
    source gets a new version row; old versions are retired, not deleted, so
    citations stored with earlier replies keep resolving."""

    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint("source_key", "version", name="uq_knowledge_documents_source_version"),
        CheckConstraint("review_status IN ('approved', 'retired')", name="review_status_valid"),
        Index(
            "uq_knowledge_documents_one_current",
            "source_key",
            unique=True,
            postgresql_where=text("review_status = 'approved'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_key: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(String(128))
    licence: Mapped[str] = mapped_column(String(128))
    licence_url: Mapped[str | None] = mapped_column(Text)
    attribution: Mapped[str | None] = mapped_column(Text)
    reviewed_on: Mapped[date | None] = mapped_column(Date)
    topics: Mapped[list[str]] = mapped_column(JSONB)
    review_status: Mapped[str] = mapped_column(String(16))
    # SHA-256 of the normalised extracted text: unchanged text, no new version.
    content_checksum: Mapped[str] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeChunk(Base):
    """A passage of a document version (spec §11.5); every chunk traces to its
    document. Deleted only with its document version."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_knowledge_chunks_document_ordinal"),
        CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)", name="embedding_has_model"
        ),
        Index("ix_knowledge_chunks_search", "search", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    # SHA-256 of the normalised chunk text, for de-duplication.
    text_checksum: Mapped[str] = mapped_column(String(64))
    # English full-text vector of heading and text, kept by the database.
    search: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', heading || ' ' || text)", persisted=True),
    )
    # Vectors from different models are never compared; dimensions vary by model.
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(128))

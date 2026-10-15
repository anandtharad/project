"""Transactional turn persistence.

A turn uses two short transactions around the (slow) model call, so no
database transaction or lock is held while generating:

1. `begin`: lock the conversation row, resolve idempotency, refuse a second
   in-flight turn, persist the user message as `pending` with a fresh
   attempt ID, and load recent history.
2. `complete`: lock the conversation row again, verify this attempt still
   owns the pending turn, allocate the next sequence for the assistant
   message, mark the turn `completed`, and store the turn's decisions and
   updated conversation state - atomically.

Conversation state uses optimistic concurrency: `begin` reads the state and
its version; `complete` writes the new state only if the stored version is
still the one read (under the conversation row lock), so a stale write never
overwrites a newer state. A rejected write leaves the reply intact.

Concurrency rules (one in-flight turn per conversation):
- Ordering comes from a per-conversation sequence counter updated only under
  the conversation row lock, never from timestamps.
- A second message while a turn is pending is rejected with 409
  `turn_in_progress`. A partial unique index enforces the same invariant in
  the database.
- A pending turn older than `stale_turn_seconds` (measured with the database
  clock) is treated as abandoned and may be taken over; the abandoned
  attempt can no longer complete because its attempt ID no longer matches.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import repositories as repo
from app.db.models import (
    Conversation,
    ConversationStateRecord,
    Message,
    MessageRole,
    TurnDecision,
    TurnStatus,
)
from app.decisions.schemas import DecisionResult
from app.errors import AppError, ErrorCategory, conversation_not_found
from app.llm.schemas import ChatMessage, ChatRole
from app.observability import log_event
from app.state.schemas import STATE_SCHEMA_VERSION, ConversationState, StoredState

logger = logging.getLogger("app.turn")


class TurnAttemptLost(Exception):
    """The attempt no longer owns the pending turn (taken over or resolved)."""


def turn_in_progress() -> AppError:
    return AppError(
        ErrorCategory.CONFLICT,
        "turn_in_progress",
        "Another message in this conversation is still being processed. Retry shortly.",
        retry_after_seconds=2,
    )


def turn_superseded() -> AppError:
    return AppError(
        ErrorCategory.CONFLICT,
        "turn_superseded",
        "This message can no longer be completed because the conversation has moved on.",
    )


def fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompletedTurn:
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    assistant_content: str
    created_at: datetime
    replayed: bool = False
    # Version of the stored state after this turn; None when the write was
    # rejected as stale, not requested, or on replay.
    state_version: int | None = None


@dataclass(frozen=True)
class StartedTurn:
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    attempt_id: uuid.UUID
    history: list[ChatMessage]
    state: StoredState | None = None


@dataclass(frozen=True)
class StateWrite:
    state: ConversationState
    # The version read at `begin` (0 when there was no state yet).
    expected_version: int
    extraction: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantProvenance:
    provider: str
    model_id: str
    prompt_id: str
    prompt_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TurnStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        stale_turn_seconds: float,
        history_limit: int,
        messages_per_minute: int = 0,
        messages_per_day: int = 0,
    ) -> None:
        self._sessions = session_factory
        self._stale_after = timedelta(seconds=stale_turn_seconds)
        self._history_limit = history_limit
        # (limit, window); a limit of 0 is off.
        self._rate_limits = [
            (limit, window)
            for limit, window in (
                (messages_per_minute, timedelta(minutes=1)),
                (messages_per_day, timedelta(days=1)),
            )
            if limit > 0
        ]

    async def begin(
        self,
        *,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        idempotency_key: str | None,
    ) -> StartedTurn | CompletedTurn:
        request_fingerprint = fingerprint(content)
        try:
            async with self._sessions() as session, session.begin():
                conversation = await repo.get_owned_conversation(
                    session, conversation_id, user_id, for_update=True
                )
                if conversation is None:
                    raise conversation_not_found()
                db_now = await session.scalar(select(func.now()))
                assert db_now is not None

                if idempotency_key is not None:
                    existing = await repo.find_by_idempotency_key(
                        session, conversation.id, idempotency_key
                    )
                    if existing is not None:
                        return await self._resume_existing(
                            session, conversation, existing, request_fingerprint, db_now
                        )

                pending = await repo.find_pending_turn(session, conversation.id)
                if pending is not None:
                    if not self._is_stale(pending, db_now):
                        raise turn_in_progress()
                    pending.turn_status = TurnStatus.FAILED
                    pending.failure_category = "abandoned"
                    await session.flush()

                # Only new messages count: an idempotent retry returned above.
                await self._enforce_rate_limits(session, user_id, db_now)
                conversation.last_sequence += 1
                conversation.updated_at = db_now
                message = Message(
                    id=uuid.uuid4(),
                    conversation_id=conversation.id,
                    sequence=conversation.last_sequence,
                    role=MessageRole.USER,
                    content=content,
                    turn_status=TurnStatus.PENDING,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    attempt_id=uuid.uuid4(),
                    attempt_started_at=db_now,
                )
                session.add(message)
                await session.flush()
                return await self._started(session, conversation, message)
        except IntegrityError as exc:
            # Backstop: the row lock should make this unreachable.
            raise turn_in_progress() from exc

    async def _resume_existing(
        self,
        session: AsyncSession,
        conversation: Conversation,
        existing: Message,
        request_fingerprint: str,
        db_now: datetime,
    ) -> StartedTurn | CompletedTurn:
        if existing.request_fingerprint != request_fingerprint:
            raise AppError(
                ErrorCategory.CONFLICT,
                "idempotency_key_reused",
                "This idempotency key was already used for a different message.",
            )
        if existing.turn_status == TurnStatus.COMPLETED:
            reply = await repo.get_reply(session, existing.id)
            if reply is None:  # pragma: no cover - guarded by the completion transaction
                raise AppError(ErrorCategory.INTERNAL_ERROR, "turn_inconsistent", "Internal error.")
            return CompletedTurn(
                conversation_id=conversation.id,
                user_message_id=existing.id,
                assistant_message_id=reply.id,
                assistant_content=reply.content,
                created_at=reply.created_at,
                replayed=True,
            )
        if existing.turn_status == TurnStatus.PENDING and not self._is_stale(existing, db_now):
            raise turn_in_progress()
        # Failed (or abandoned) turn: retry generation for the same user
        # message, but only while it is still the latest message.
        if existing.sequence != conversation.last_sequence:
            raise turn_superseded()
        existing.turn_status = TurnStatus.PENDING
        existing.failure_category = None
        existing.attempt_id = uuid.uuid4()
        existing.attempt_started_at = db_now
        await session.flush()
        return await self._started(session, conversation, existing)

    async def _started(
        self, session: AsyncSession, conversation: Conversation, message: Message
    ) -> StartedTurn:
        assert message.attempt_id is not None
        rows = await repo.recent_messages(
            session, conversation.id, up_to_sequence=message.sequence, limit=self._history_limit
        )
        history = [
            ChatMessage(
                role=ChatRole.USER if row.role == MessageRole.USER else ChatRole.ASSISTANT,
                content=row.content,
            )
            for row in rows
        ]
        return StartedTurn(
            conversation_id=conversation.id,
            user_message_id=message.id,
            attempt_id=message.attempt_id,
            history=history,
            state=await self._load_state(session, conversation.id),
        )

    async def _load_state(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> StoredState | None:
        record = await session.get(ConversationStateRecord, conversation_id)
        if record is None:
            return None
        try:
            state = ConversationState.model_validate(record.state)
        except ValidationError:
            # An unreadable state (e.g. an older schema) is not fatal: the turn
            # proceeds without it and the next write replaces it.
            state = ConversationState()
        return StoredState(state=state, version=record.state_version, updated_at=record.updated_at)

    async def delete_conversation(self, *, user_id: uuid.UUID, conversation_id: uuid.UUID) -> bool:
        """Delete an owned conversation with its messages, state, decisions and
        other per-turn rows (ON DELETE CASCADE). Memories are kept: their
        conversation and source-message links become NULL, and the person
        deletes memories separately. False when the conversation does not
        exist or belongs to someone else (the caller answers 404 either way).
        Refused while a turn is in flight, under the same row lock `begin`
        takes, so a reply is never written into a deleted conversation.
        """
        async with self._sessions() as session, session.begin():
            conversation = await repo.get_owned_conversation(
                session, conversation_id, user_id, for_update=True
            )
            if conversation is None:
                return False
            pending = await repo.find_pending_turn(session, conversation.id)
            if pending is not None:
                db_now = await session.scalar(select(func.now()))
                assert db_now is not None
                if not self._is_stale(pending, db_now):
                    raise turn_in_progress()
            await session.delete(conversation)
        return True

    async def _enforce_rate_limits(
        self, session: AsyncSession, user_id: uuid.UUID, db_now: datetime
    ) -> None:
        # Concurrent first messages in two different conversations can both
        # pass (each holds only its own conversation lock); the limit is a
        # guard against runaway use, not an exact quota.
        for limit, window in self._rate_limits:
            count, oldest = await repo.user_messages_since(session, user_id, db_now - window)
            if count >= limit:
                frees_at = (oldest or db_now) + window
                retry = max(1, math.ceil((frees_at - db_now).total_seconds()))
                log_event(logger, "turn.rate_limited", window_seconds=int(window.total_seconds()))
                raise AppError(
                    ErrorCategory.RATE_LIMITED,
                    "too_many_messages",
                    "You've sent a lot of messages in a short time. Please wait a moment.",
                    retry_after_seconds=retry,
                )

    def _is_stale(self, message: Message, db_now: datetime) -> bool:
        started = message.attempt_started_at
        return started is None or db_now - started > self._stale_after

    async def complete(
        self,
        *,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        attempt_id: uuid.UUID,
        content: str,
        provenance: AssistantProvenance,
        state_write: StateWrite | None = None,
        decisions: Sequence[DecisionResult] = (),
    ) -> CompletedTurn:
        async with self._sessions() as session, session.begin():
            conversation = await session.get(Conversation, conversation_id, with_for_update=True)
            user_message = await session.get(Message, user_message_id, with_for_update=True)
            if (
                conversation is None
                or user_message is None
                or user_message.conversation_id != conversation.id
                or user_message.turn_status != TurnStatus.PENDING
                or user_message.attempt_id != attempt_id
            ):
                raise TurnAttemptLost()
            conversation.last_sequence += 1
            conversation.updated_at = func.now()
            assistant = Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                sequence=conversation.last_sequence,
                role=MessageRole.ASSISTANT,
                content=content,
                reply_to_message_id=user_message.id,
                provider=provenance.provider,
                model_id=provenance.model_id,
                prompt_id=provenance.prompt_id,
                prompt_version=provenance.prompt_version,
                generation_metadata=provenance.metadata,
            )
            user_message.turn_status = TurnStatus.COMPLETED
            user_message.failure_category = None
            session.add(assistant)
            session.add_all(
                _decision_row(conversation.id, user_message.id, result) for result in decisions
            )
            state_version = (
                await self._write_state(session, conversation.id, user_message.id, state_write)
                if state_write is not None
                else None
            )
            await session.flush()
            return CompletedTurn(
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant.id,
                assistant_content=assistant.content,
                created_at=assistant.created_at,
                state_version=state_version,
            )

    async def _write_state(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        source_message_id: uuid.UUID,
        write: StateWrite,
    ) -> int | None:
        """Called under the conversation row lock. Returns the new version, or
        None when the write is stale (another turn stored a newer state)."""
        record = await session.get(ConversationStateRecord, conversation_id, with_for_update=True)
        current = record.state_version if record is not None else 0
        if current != write.expected_version:
            return None
        data = write.state.model_dump(mode="json")
        if record is None:
            record = ConversationStateRecord(
                conversation_id=conversation_id,
                schema_version=STATE_SCHEMA_VERSION,
                state=data,
                state_version=1,
                source_message_id=source_message_id,
                extraction=write.extraction,
            )
            session.add(record)
        else:
            record.schema_version = STATE_SCHEMA_VERSION
            record.state = data
            record.state_version = current + 1
            record.source_message_id = source_message_id
            record.extraction = write.extraction
            record.updated_at = func.now()
        return current + 1

    async def fail(
        self, *, user_message_id: uuid.UUID, attempt_id: uuid.UUID, category: str
    ) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Message)
                .where(
                    Message.id == user_message_id,
                    Message.attempt_id == attempt_id,
                    Message.turn_status == TurnStatus.PENDING,
                )
                .values(turn_status=TurnStatus.FAILED, failure_category=category)
            )


def _decision_row(
    conversation_id: uuid.UUID, user_message_id: uuid.UUID, result: DecisionResult
) -> TurnDecision:
    return TurnDecision(
        id=result.decision_id,
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        question_id=result.question_id,
        question_version=result.question_version,
        kind=result.kind,
        value=result.value,
        returned_value=result.returned_value,
        probabilities=result.probabilities,
        confidence=result.confidence,
        provider=result.provider,
        model_id=result.model_id,
        latency_ms=result.latency_ms,
        validation_status=result.validation_status.value,
        fallback=result.fallback,
        fallback_reason=result.fallback_reason.value if result.fallback_reason else None,
    )

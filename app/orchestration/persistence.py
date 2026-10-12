"""Transactional turn persistence.

A turn uses two short transactions around the (slow) model call, so no
database transaction or lock is held while generating:

1. `begin`: lock the conversation row, resolve idempotency, refuse a second
   in-flight turn, persist the user message as `pending` with a fresh
   attempt ID, and load recent history.
2. `complete`: lock the conversation row again, verify this attempt still
   owns the pending turn, allocate the next sequence for the assistant
   message, and mark the turn `completed` - atomically.

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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import repositories as repo
from app.db.models import Conversation, Message, MessageRole, TurnStatus
from app.errors import AppError, ErrorCategory, conversation_not_found
from app.llm.schemas import ChatMessage, ChatRole


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


@dataclass(frozen=True)
class StartedTurn:
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    attempt_id: uuid.UUID
    history: list[ChatMessage]


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
    ) -> None:
        self._sessions = session_factory
        self._stale_after = timedelta(seconds=stale_turn_seconds)
        self._history_limit = history_limit

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
            await session.flush()
            return CompletedTurn(
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant.id,
                assistant_content=assistant.content,
                created_at=assistant.created_at,
            )

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

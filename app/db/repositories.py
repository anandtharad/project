"""Query helpers. Every conversation lookup that starts from a client request
is scoped by owner in the query itself (`get_owned_conversation`); message
queries take a conversation ID that has already passed that check."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, TurnStatus, User


async def get_or_create_user(session: AsyncSession, external_id: str) -> uuid.UUID:
    inserted = await session.scalar(
        pg_insert(User)
        .values(id=uuid.uuid4(), external_id=external_id)
        .on_conflict_do_nothing(index_elements=[User.external_id])
        .returning(User.id)
    )
    if isinstance(inserted, uuid.UUID):
        return inserted
    existing = await session.scalar(select(User.id).where(User.external_id == external_id))
    if existing is None:  # pragma: no cover - would require a concurrent delete
        raise RuntimeError("user row vanished during get_or_create")
    return existing


async def create_conversation(session: AsyncSession, user_id: uuid.UUID) -> Conversation:
    conversation = Conversation(id=uuid.uuid4(), user_id=user_id)
    session.add(conversation)
    await session.flush()
    return conversation


async def get_owned_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    for_update: bool = False,
) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    )
    if for_update:
        stmt = stmt.with_for_update()
    return await session.scalar(stmt)


async def count_messages(session: AsyncSession, conversation_id: uuid.UUID) -> int:
    count = await session.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
    )
    return int(count or 0)


async def find_by_idempotency_key(
    session: AsyncSession, conversation_id: uuid.UUID, key: str
) -> Message | None:
    return await session.scalar(
        select(Message).where(
            Message.conversation_id == conversation_id, Message.idempotency_key == key
        )
    )


async def find_pending_turn(session: AsyncSession, conversation_id: uuid.UUID) -> Message | None:
    return await session.scalar(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.turn_status == TurnStatus.PENDING,
        )
    )


async def get_reply(session: AsyncSession, user_message_id: uuid.UUID) -> Message | None:
    return await session.scalar(
        select(Message).where(Message.reply_to_message_id == user_message_id)
    )


async def recent_messages(
    session: AsyncSession, conversation_id: uuid.UUID, *, up_to_sequence: int, limit: int
) -> list[Message]:
    """The `limit` most recent messages up to a sequence, oldest first."""
    rows = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.sequence <= up_to_sequence)
        .order_by(Message.sequence.desc())
        .limit(limit)
    )
    return list(reversed(rows.all()))


async def page_messages(
    session: AsyncSession, conversation_id: uuid.UUID, *, after_sequence: int, limit: int
) -> list[Message]:
    rows = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.sequence > after_sequence)
        .order_by(Message.sequence)
        .limit(limit)
    )
    return list(rows.all())

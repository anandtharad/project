"""Memory persistence and retrieval (spec §10.4-10.7).

Every query is scoped by `user_id` in SQL: a memory ID alone never reaches
another person's data. Deleting a memory deletes its row, and the embedding
lives in that row, so no derived copy remains in the database.

Retrieval is hybrid and deliberately small:
- semantic: cosine similarity between the query and memory embeddings, only
  between vectors from the same embedding model;
- lexical: PostgreSQL full-text match (`english` configuration, any word),
  which boosts semantic matches and is the fallback when no embedding is
  available (spec §23);
- only memories with the requested statuses (active, and pending ones shown
  as unconfirmed) that have not expired and are within the retention period
  are candidates; results are thresholded, de-duplicated and capped at
  `limit`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    Float,
    Text,
    and_,
    case,
    cast,
    delete,
    func,
    literal,
    not_,
    null,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import defer

from app.db.models import MemoryRecord, Message, UserMemorySettings
from app.memory.policy import MemorySettings, PolicyDecision, looks_financial, normalize
from app.memory.schemas import (
    MEMORY_SCHEMA_VERSION,
    Memory,
    MemoryStatus,
    MemoryType,
    RetrievedMemory,
)

# Candidates considered per retrieval, best-scored first.
_CANDIDATE_POOL = 500


@dataclass(frozen=True)
class RetrievalParams:
    limit: int = 5
    # Minimum cosine similarity for a semantic match.
    min_semantic: float = 0.6
    # Whether a full-text match alone qualifies a memory that has a vector.
    # Without vectors (no embeddings, or a failed call) it always does.
    lexical_only_ok: bool = False


@dataclass(frozen=True)
class NewMemory:
    decision: PolicyDecision
    embedding: list[float] | None
    embedding_model: str | None


def to_memory(record: MemoryRecord) -> Memory:
    return Memory(
        id=record.id,
        memory_type=MemoryType(record.memory_type),
        content=record.content,
        status=MemoryStatus(record.status),
        importance=record.importance,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        source_conversation_id=record.source_conversation_id,
        supersedes_id=record.supersedes_id,
        provenance=record.provenance,
    )


def select_hits(
    candidates: Iterable[RetrievedMemory], params: RetrievalParams
) -> list[RetrievedMemory]:
    """Threshold, de-duplicate and rank scored candidates (given newest first,
    so a duplicate keeps its newest copy). Shared with eval/memory runs."""
    scored: list[RetrievedMemory] = []
    seen: set[str] = set()
    for candidate in candidates:
        sem, lex = candidate.semantic, candidate.lexical
        semantic_hit = sem is not None and sem >= params.min_semantic
        lexical_hit = lex > 0 and (params.lexical_only_ok or sem is None)
        if not (semantic_hit or lexical_hit):
            continue
        key = normalize(candidate.memory.content)
        if key in seen:
            continue
        seen.add(key)
        # Semantic similarity leads; a full-text match adds a small boost.
        score = (sem or 0.0) + min(lex, 1.0) * 0.2
        scored.append(candidate.model_copy(update={"score": score}))
    scored.sort(key=lambda r: (r.score, r.memory.created_at), reverse=True)
    return scored[: params.limit]


async def _lock_settings(session: AsyncSession, user_id: uuid.UUID) -> UserMemorySettings:
    """The person's settings row, created with defaults if missing, locked for
    the rest of the transaction (settings changes and delete-all wait)."""
    await session.execute(
        pg_insert(UserMemorySettings)
        .values(user_id=user_id)
        .on_conflict_do_nothing(index_elements=[UserMemorySettings.user_id])
    )
    row = await session.scalar(
        select(UserMemorySettings)
        .where(UserMemorySettings.user_id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    assert row is not None
    return row


async def _supersede(session: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> None:
    """Mark a memory superseded, together with any other pending proposal to
    replace it: once the question is settled, those are out of date too."""
    await session.execute(
        update(MemoryRecord)
        .where(
            MemoryRecord.user_id == user_id,
            or_(
                MemoryRecord.id == memory_id,
                (MemoryRecord.supersedes_id == memory_id)
                & (MemoryRecord.status == MemoryStatus.PENDING_CONFIRMATION),
            ),
        )
        .values(status=MemoryStatus.SUPERSEDED, updated_at=func.now())
    )


class MemoryStore:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, retention_days: int = 365
    ) -> None:
        self._sessions = session_factory
        # Applied to existing memories too, so lowering it takes effect at once.
        self._retention = timedelta(days=retention_days)

    def _live(self, now: datetime) -> Any:
        """Not expired, and within the retention period."""
        return and_(
            or_(MemoryRecord.expires_at.is_(None), MemoryRecord.expires_at > now),
            MemoryRecord.created_at > now - self._retention,
        )

    # --- user controls --------------------------------------------------------

    async def list_memories(
        self, user_id: uuid.UUID, *, statuses: Sequence[MemoryStatus], now: datetime
    ) -> list[Memory]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(MemoryRecord)
                .options(defer(MemoryRecord.embedding))
                .where(
                    MemoryRecord.user_id == user_id,
                    MemoryRecord.status.in_([s.value for s in statuses]),
                    self._live(now),
                )
                .order_by(MemoryRecord.created_at.desc())
            )
            return [to_memory(r) for r in rows]

    async def get(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> Memory | None:
        async with self._sessions() as session:
            record = await session.scalar(
                select(MemoryRecord).where(
                    MemoryRecord.id == memory_id,
                    MemoryRecord.user_id == user_id,
                    self._live(datetime.now(UTC)),
                )
            )
            return to_memory(record) if record is not None else None

    async def delete(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(MemoryRecord)
                .where(MemoryRecord.id == memory_id, MemoryRecord.user_id == user_id)
                .returning(MemoryRecord.id)
            )
            return result.first() is not None

    async def delete_all(self, user_id: uuid.UUID) -> int:
        """Delete every memory and record when, so an extraction already
        running for an earlier message cannot store anything afterwards."""
        async with self._sessions() as session, session.begin():
            await _lock_settings(session, user_id)
            await session.execute(
                update(UserMemorySettings)
                .where(UserMemorySettings.user_id == user_id)
                .values(memories_cleared_at=func.now())
            )
            result = await session.execute(
                delete(MemoryRecord)
                .where(MemoryRecord.user_id == user_id)
                .returning(MemoryRecord.id)
            )
            return len(result.all())

    async def confirm(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> Memory | None:
        """A pending memory becomes active and supersedes the memory it conflicted with."""
        async with self._sessions() as session, session.begin():
            # Every per-person write takes this lock first, so writes queue in
            # one order instead of deadlocking on memory rows.
            await _lock_settings(session, user_id)
            record = await session.scalar(
                select(MemoryRecord)
                .where(
                    MemoryRecord.id == memory_id,
                    MemoryRecord.user_id == user_id,
                    self._live(datetime.now(UTC)),
                )
                .with_for_update()
            )
            if record is None:
                return None
            if record.status == MemoryStatus.PENDING_CONFIRMATION:
                record.status = MemoryStatus.ACTIVE
                record.updated_at = func.now()
                await session.flush()
                if record.supersedes_id is not None:
                    await _supersede(session, user_id, record.supersedes_id)
                await session.refresh(record)
            return to_memory(record)

    async def get_settings(self, user_id: uuid.UUID) -> MemorySettings:
        async with self._sessions() as session:
            row = await session.get(UserMemorySettings, user_id)
            if row is None:
                return MemorySettings()
            return MemorySettings(
                memory_enabled=row.memory_enabled, allow_sensitive=row.allow_sensitive
            )

    async def put_settings(self, user_id: uuid.UUID, settings: MemorySettings) -> MemorySettings:
        async with self._sessions() as session, session.begin():
            await session.execute(
                pg_insert(UserMemorySettings)
                .values(
                    user_id=user_id,
                    memory_enabled=settings.memory_enabled,
                    allow_sensitive=settings.allow_sensitive,
                )
                .on_conflict_do_update(
                    index_elements=[UserMemorySettings.user_id],
                    set_={
                        "memory_enabled": settings.memory_enabled,
                        "allow_sensitive": settings.allow_sensitive,
                        "updated_at": func.now(),
                    },
                )
            )
        return settings

    # --- writes from post-turn processing ------------------------------------

    async def apply(
        self,
        user_id: uuid.UUID,
        *,
        conversation_id: uuid.UUID,
        source_message_id: uuid.UUID,
        new: Sequence[NewMemory],
        provenance: dict[str, Any],
    ) -> list[uuid.UUID]:
        """Store accepted memories in one transaction; returns their IDs.

        The person's settings may have changed while extraction was running.
        They are read again here under a row lock (the row is created if it
        does not exist yet, so the lock always holds): nothing is stored if
        memory was turned off or all memories were deleted after the source
        message was sent, and nothing sensitive if `allow_sensitive` was
        turned off."""
        created: list[uuid.UUID] = []
        async with self._sessions() as session, session.begin():
            settings = await _lock_settings(session, user_id)
            if not settings.memory_enabled:
                return []
            if settings.memories_cleared_at is not None:
                sent = await session.scalar(
                    select(Message.created_at).where(Message.id == source_message_id)
                )
                if sent is None or sent <= settings.memories_cleared_at:
                    return []
            if not settings.allow_sensitive:
                new = [
                    item
                    for item in new
                    if item.decision.candidate.sensitive_category == "none"
                    and not looks_financial(item.decision.content)
                ]
            # A memory to supersede or conflict with may have been deleted (or
            # purged) while extraction ran: then this is just a new memory.
            related = {item.decision.related_id for item in new} - {None}
            existing = (
                set(
                    await session.scalars(
                        select(MemoryRecord.id)
                        .where(MemoryRecord.user_id == user_id, MemoryRecord.id.in_(related))
                        .with_for_update()
                    )
                )
                if related
                else set()
            )
            superseded: list[uuid.UUID] = []
            for item in new:
                d = item.decision
                if d.related_id is not None and d.related_id not in existing:
                    d = replace(d, action="store", related_id=None)
                assert d.status is not None
                memory_id = uuid.uuid4()
                session.add(
                    MemoryRecord(
                        id=memory_id,
                        user_id=user_id,
                        memory_type=d.candidate.memory_type.value,
                        content=d.content,
                        status=d.status.value,
                        importance=d.importance,
                        # The supporting quote is checked but not kept: it can
                        # hold more than the memory (source_message_id points
                        # at the message).
                        provenance={
                            **provenance,
                            "explicit": d.candidate.explicit,
                            "sensitive_category": d.candidate.sensitive_category,
                            "policy": d.reason,
                            "lifetime": d.candidate.lifetime,
                        },
                        source_conversation_id=conversation_id,
                        source_message_id=source_message_id,
                        supersedes_id=d.related_id,
                        expires_at=d.expires_at,
                        embedding=item.embedding,
                        embedding_model=item.embedding_model if item.embedding else None,
                        schema_version=MEMORY_SCHEMA_VERSION,
                    )
                )
                if d.action == "supersede" and d.related_id is not None:
                    superseded.append(d.related_id)
                created.append(memory_id)
            # After every row is in, so a conflicting proposal from this same
            # batch is superseded too, whatever the order.
            await session.flush()
            for memory_id in superseded:
                await _supersede(session, user_id, memory_id)
        return created

    async def purge_expired(self, now: datetime) -> int:
        """Retention: delete memories past their expiry or older than the
        retention period, for every user."""
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(MemoryRecord).where(not_(self._live(now))).returning(MemoryRecord.id)
            )
            return len(result.all())

    async def needing_embedding(
        self, embedding_model: str, *, limit: int
    ) -> list[tuple[uuid.UUID, str]]:
        """Maintenance, across users: memories without a vector from this model."""
        async with self._sessions() as session:
            rows = await session.execute(
                select(MemoryRecord.id, MemoryRecord.content)
                .where(
                    MemoryRecord.status != MemoryStatus.SUPERSEDED,
                    or_(
                        MemoryRecord.embedding_model.is_(None),
                        MemoryRecord.embedding_model != embedding_model,
                    ),
                )
                .order_by(MemoryRecord.created_at.desc())
                .limit(limit)
            )
            return [(memory_id, content) for memory_id, content in rows]

    async def set_embeddings(
        self, items: Sequence[tuple[uuid.UUID, list[float]]], embedding_model: str
    ) -> int:
        updated = 0
        async with self._sessions() as session, session.begin():
            for memory_id, vector in items:
                result = await session.execute(
                    update(MemoryRecord)
                    .where(MemoryRecord.id == memory_id)
                    .values(embedding=vector, embedding_model=embedding_model)
                    .returning(MemoryRecord.id)
                )
                updated += result.first() is not None
        return updated

    # --- retrieval -----------------------------------------------------------

    async def retrieve(
        self,
        user_id: uuid.UUID,
        *,
        query: str,
        query_embedding: list[float] | None,
        embedding_model: str | None,
        now: datetime,
        params: RetrievalParams,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
    ) -> list[RetrievedMemory]:
        # Any word may match: plainto_tsquery joins words with AND, which a
        # conversational message almost never satisfies, so its operators are
        # turned into OR ('a' & 'b' -> 'a' | 'b'; it emits no other operator).
        tsquery = cast(
            func.replace(cast(func.plainto_tsquery("english", query), Text), "&", "|"), TSQUERY
        )
        lexical = func.ts_rank_cd(func.to_tsvector("english", MemoryRecord.content), tsquery)
        # Only vectors from the same model are compared (dimensions differ);
        # memories embedded by another model, or not at all, can still match
        # by full text. CASE evaluates the distance only for matching rows.
        semantic = (
            case(
                (
                    MemoryRecord.embedding_model == embedding_model,
                    1 - MemoryRecord.embedding.cosine_distance(query_embedding),
                ),
                else_=null(),
            ).label("semantic")
            if query_embedding is not None and embedding_model is not None
            else literal(None, type_=Float).label("semantic")
        )
        conditions = [
            MemoryRecord.user_id == user_id,
            MemoryRecord.status.in_([s.value for s in statuses]),
            self._live(now),
        ]
        # The pool is ranked by the same score select_hits uses, so a large
        # memory set never hides a strong match behind newer, weaker ones.
        score = func.coalesce(semantic, 0.0) + func.least(lexical, 1.0) * 0.2
        async with self._sessions() as session:
            stmt = (
                select(MemoryRecord, semantic, lexical.label("lexical"))
                .options(defer(MemoryRecord.embedding))
                .where(*conditions)
                .order_by(score.desc(), MemoryRecord.created_at.desc())
                .limit(_CANDIDATE_POOL)
            )
            rows = (await session.execute(stmt)).all()
        return select_hits(
            (
                RetrievedMemory(
                    memory=to_memory(record),
                    score=0.0,
                    semantic=float(sem) if sem is not None else None,
                    lexical=float(lex or 0.0),
                )
                for record, sem, lex in rows
            ),
            params,
        )

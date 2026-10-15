"""Knowledge persistence (spec §11.3-11.6).

Documents are versioned: re-ingesting an unchanged source is a no-op, a
changed one gets a new version and the old version is retired (kept, so
citations stored with earlier replies still resolve). Only chunks of
approved (current) versions are searched.

Search has two candidate lists, merged by app/rag/retrieval.py:
- lexical: PostgreSQL full-text match on the generated `search` vector of
  heading and text (`english` configuration, any word);
- semantic: exact cosine similarity, only between vectors of the same
  embedding model. The embedding column has no fixed dimension (it depends
  on the model), so it has no approximate index; an exact scan is fine for
  a corpus of this size (hundreds of chunks).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from sqlalchemy import Float, Text, cast, func, literal, select, text, update
from sqlalchemy.dialects.postgresql import JSONB, TSQUERY, array
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import defer

from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.rag.chunking import Chunk

APPROVED = "approved"
RETIRED = "retired"


@dataclass(frozen=True)
class SourceMetadata:
    """A source's provenance as recorded in the manifest (spec §11.3)."""

    key: str
    url: str
    title: str
    publisher: str
    licence: str
    licence_url: str | None
    attribution: str | None
    published_on: date | None
    reviewed_on: date
    topics: tuple[str, ...]

    def columns(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "publisher": self.publisher,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "attribution": self.attribution,
            "published_on": self.published_on,
            "reviewed_on": self.reviewed_on,
            "topics": list(self.topics),
        }


class IngestOutcome(StrEnum):
    CREATED = "created"
    NEW_VERSION = "new_version"
    UNCHANGED = "unchanged"
    METADATA_UPDATED = "metadata_updated"
    # Same text as the current version of another source: not stored twice.
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class IngestResult:
    outcome: IngestOutcome
    document_id: uuid.UUID | None
    version: int | None
    chunks_written: int = 0
    duplicate_chunks: int = 0
    duplicate_of: str | None = None


@dataclass(frozen=True)
class KnowledgeFilter:
    """Source filtering (spec §11.6): empty means no restriction."""

    topics: tuple[str, ...] = ()
    source_keys: tuple[str, ...] = ()
    publishers: tuple[str, ...] = ()


NO_FILTER = KnowledgeFilter()


@dataclass(frozen=True)
class Candidate:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_key: str
    version: int
    title: str
    publisher: str
    url: str
    reviewed_on: date | None
    heading: str
    text: str
    text_checksum: str
    semantic: float | None = None
    lexical: float | None = None


@dataclass(frozen=True)
class CitationTarget:
    """What a citation resolves to (spec §11.8), including retired versions."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_key: str
    version: int
    title: str
    publisher: str
    url: str
    heading: str
    current: bool


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    retired_documents: int
    chunks: int
    embedded: dict[str, int] = field(default_factory=dict)


def _or_tsquery(query: str) -> Any:
    # plainto_tsquery joins words with AND, which a question rarely satisfies
    # in full; its operators are turned into OR (it emits no other operator).
    return cast(func.replace(cast(func.plainto_tsquery("english", query), Text), "&", "|"), TSQUERY)


class KnowledgeStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # --- ingestion ---------------------------------------------------------

    async def upsert_document(
        self, meta: SourceMetadata, checksum: str, chunks: Sequence[Chunk]
    ) -> IngestResult:
        async with self._sessions() as session, session.begin():
            # One ingestion per source at a time (two runs cannot both add a version).
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": f"knowledge:{meta.key}"}
            )
            current = await session.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.source_key == meta.key,
                    KnowledgeDocument.review_status == APPROVED,
                )
            )
            if current is not None and current.content_checksum == checksum:
                columns = meta.columns()
                changed = {k: v for k, v in columns.items() if getattr(current, k) != v}
                if not changed:
                    return IngestResult(IngestOutcome.UNCHANGED, current.id, current.version)
                for name, value in changed.items():
                    setattr(current, name, value)
                return IngestResult(IngestOutcome.METADATA_UPDATED, current.id, current.version)

            twin = await session.scalar(
                select(KnowledgeDocument.source_key).where(
                    KnowledgeDocument.content_checksum == checksum,
                    KnowledgeDocument.review_status == APPROVED,
                    KnowledgeDocument.source_key != meta.key,
                )
            )
            if twin is not None:
                return IngestResult(IngestOutcome.DUPLICATE, None, None, duplicate_of=twin)

            latest = await session.scalar(
                select(func.max(KnowledgeDocument.version)).where(
                    KnowledgeDocument.source_key == meta.key
                )
            )
            if current is not None:
                current.review_status = RETIRED
                current.retired_at = func.now()
                # The partial unique index allows one approved version per key.
                await session.flush()
            document = KnowledgeDocument(
                source_key=meta.key,
                version=(latest or 0) + 1,
                review_status=APPROVED,
                content_checksum=checksum,
                **meta.columns(),
            )
            session.add(document)
            await session.flush()
            seen: set[str] = set()
            written = 0
            for chunk in chunks:
                if chunk.checksum in seen:
                    continue
                seen.add(chunk.checksum)
                session.add(
                    KnowledgeChunk(
                        document_id=document.id,
                        ordinal=written,
                        heading=chunk.heading,
                        text=chunk.text,
                        text_checksum=chunk.checksum,
                    )
                )
                written += 1
            outcome = IngestOutcome.NEW_VERSION if current is not None else IngestOutcome.CREATED
            return IngestResult(
                outcome,
                document.id,
                document.version,
                chunks_written=written,
                duplicate_chunks=len(chunks) - written,
            )

    async def retire_missing(self, keep: Iterable[str]) -> list[str]:
        """Retire approved sources whose key is not in `keep` (a source removed
        from the manifest). Returns the retired keys."""
        async with self._sessions() as session, session.begin():
            rows = await session.scalars(
                update(KnowledgeDocument)
                .where(
                    KnowledgeDocument.review_status == APPROVED,
                    KnowledgeDocument.source_key.not_in(list(keep)),
                )
                .values(review_status=RETIRED, retired_at=func.now())
                .returning(KnowledgeDocument.source_key)
            )
            return sorted(rows)

    async def chunks_needing_embedding(
        self, embedding_model: str, *, limit: int
    ) -> list[tuple[uuid.UUID, str, str]]:
        """(chunk id, heading, text) of current chunks with no vector from this model."""
        async with self._sessions() as session:
            rows = await session.execute(
                select(KnowledgeChunk.id, KnowledgeChunk.heading, KnowledgeChunk.text)
                .join(KnowledgeDocument)
                .where(
                    KnowledgeDocument.review_status == APPROVED,
                    KnowledgeChunk.embedding_model.is_distinct_from(embedding_model),
                )
                .order_by(KnowledgeChunk.document_id, KnowledgeChunk.ordinal)
                .limit(limit)
            )
            return [(r[0], r[1], r[2]) for r in rows]

    async def set_embeddings(
        self, items: Sequence[tuple[uuid.UUID, list[float]]], embedding_model: str
    ) -> None:
        async with self._sessions() as session, session.begin():
            for chunk_id, vector in items:
                await session.execute(
                    update(KnowledgeChunk)
                    .where(KnowledgeChunk.id == chunk_id)
                    .values(embedding=vector, embedding_model=embedding_model)
                )

    async def stats(self) -> CorpusStats:
        async with self._sessions() as session:
            status_rows = await session.execute(
                select(KnowledgeDocument.review_status, func.count()).group_by(
                    KnowledgeDocument.review_status
                )
            )
            # Not dict(result): a Result has .keys() (its column names).
            counts = {status: int(n) for status, n in status_rows}
            chunk_rows = await session.execute(
                select(KnowledgeChunk.embedding_model, func.count())
                .join(KnowledgeDocument)
                .where(KnowledgeDocument.review_status == APPROVED)
                .group_by(KnowledgeChunk.embedding_model)
            )
            per_model: dict[str | None, int] = {model: int(n) for model, n in chunk_rows}
        embedded = {m: n for m, n in per_model.items() if m is not None}
        return CorpusStats(
            documents=counts.get(APPROVED, 0),
            retired_documents=counts.get(RETIRED, 0),
            chunks=sum(per_model.values()),
            embedded=embedded,
        )

    # --- search --------------------------------------------------------------

    @staticmethod
    def _conditions(filters: KnowledgeFilter) -> list[Any]:
        conditions: list[Any] = [KnowledgeDocument.review_status == APPROVED]
        if filters.topics:
            conditions.append(
                KnowledgeDocument.topics.op("?|")(array(list(filters.topics)))  # any topic
            )
        if filters.source_keys:
            conditions.append(KnowledgeDocument.source_key.in_(list(filters.source_keys)))
        if filters.publishers:
            conditions.append(KnowledgeDocument.publisher.in_(list(filters.publishers)))
        return conditions

    async def _search(
        self, score: Any, order: Any, conditions: Sequence[Any], limit: int, kind: str
    ) -> list[Candidate]:
        async with self._sessions() as session:
            rows = await session.execute(
                select(KnowledgeChunk, KnowledgeDocument, score)
                .join(KnowledgeDocument)
                .options(defer(KnowledgeChunk.embedding))
                .where(*conditions)
                # Content, not ids, breaks ties: the same corpus, the same order.
                .order_by(order, KnowledgeChunk.text_checksum)
                .limit(limit)
            )
            return [
                Candidate(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    source_key=document.source_key,
                    version=document.version,
                    title=document.title,
                    publisher=document.publisher,
                    url=document.url,
                    reviewed_on=document.reviewed_on,
                    heading=chunk.heading,
                    text=chunk.text,
                    text_checksum=chunk.text_checksum,
                    semantic=float(value) if kind == "semantic" else None,
                    lexical=float(value) if kind == "lexical" else None,
                )
                for chunk, document, value in rows
            ]

    async def lexical(self, query: str, *, limit: int, filters: KnowledgeFilter) -> list[Candidate]:
        tsquery = _or_tsquery(query)
        # Normalization 32: rank / (rank + 1), so scores fall in [0, 1).
        rank = func.ts_rank_cd(KnowledgeChunk.search, tsquery, 32).label("lexical")
        conditions = [*self._conditions(filters), KnowledgeChunk.search.op("@@")(tsquery)]
        return await self._search(rank, rank.desc(), conditions, limit, "lexical")

    async def semantic(
        self, vector: list[float], embedding_model: str, *, limit: int, filters: KnowledgeFilter
    ) -> list[Candidate]:
        distance = KnowledgeChunk.embedding.cosine_distance(vector)
        similarity = (literal(1.0, type_=Float) - distance).label("semantic")
        conditions = [
            *self._conditions(filters),
            KnowledgeChunk.embedding_model == embedding_model,
        ]
        return await self._search(similarity, distance.asc(), conditions, limit, "semantic")

    async def resolve(self, chunk_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, CitationTarget]:
        ids = list(set(chunk_ids))
        if not ids:
            return {}
        async with self._sessions() as session:
            rows = await session.execute(
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.heading,
                    KnowledgeDocument.id,
                    KnowledgeDocument.source_key,
                    KnowledgeDocument.version,
                    KnowledgeDocument.title,
                    KnowledgeDocument.publisher,
                    KnowledgeDocument.url,
                    KnowledgeDocument.review_status,
                )
                .join(KnowledgeDocument)
                .where(KnowledgeChunk.id.in_(ids))
            )
            return {
                r[0]: CitationTarget(
                    chunk_id=r[0],
                    heading=r[1],
                    document_id=r[2],
                    source_key=r[3],
                    version=r[4],
                    title=r[5],
                    publisher=r[6],
                    url=r[7],
                    current=r[8] == APPROVED,
                )
                for r in rows
            }

    async def topics(self) -> set[str]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(func.jsonb_array_elements_text(cast(KnowledgeDocument.topics, JSONB)))
                .where(KnowledgeDocument.review_status == APPROVED)
                .distinct()
            )
            return set(rows)

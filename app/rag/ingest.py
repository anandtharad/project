"""Knowledge ingestion pipeline (spec §11.4), from manifest and snapshots.

    snapshot (acquired and parsed by scripts/fetch_knowledge.py)
      -> normalization -> metadata validation (manifest) -> deduplication
      -> chunking -> versioned storage (full-text vector kept by PostgreSQL)
      -> embeddings -> retrieval checks (each source's probe queries)

Idempotent: a source whose normalized text and chunking settings are
unchanged is not re-chunked (only changed metadata is updated), so
re-ingestion never duplicates chunks. A changed source gets a new version;
the previous version is retired, not deleted. Every step is counted in the
report, and a failing source never stops the others.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.rag.chunking import ChunkParams, chunk_page, embedding_text
from app.rag.extract import Block, normalize_text
from app.rag.manifest import Manifest, SnapshotError, SourceEntry, read_snapshot
from app.rag.retrieval import KnowledgeRetriever
from app.rag.store import NO_FILTER, CorpusStats, IngestOutcome, KnowledgeStore

# A source with less text than this is almost certainly a failed extraction.
MIN_DOCUMENT_WORDS = 60
EMBED_BATCH = 32


@dataclass(frozen=True)
class SourceReport:
    key: str
    outcome: str  # an IngestOutcome value, or "failed"
    version: int | None = None
    words: int = 0
    chunks: int = 0
    duplicate_chunks: int = 0
    duplicate_of: str | None = None
    failure: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    key: str
    query: str
    found: bool
    rank: int | None
    mode: str | None


@dataclass
class IngestReport:
    started_at: str
    manifest: str
    chunking: str
    sources: list[SourceReport] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    embedding_model: str | None = None
    embedded: int = 0
    embedding_failure: str | None = None
    probes: list[ProbeResult] = field(default_factory=list)
    corpus: CorpusStats | None = None
    seconds: float = 0.0

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(s.outcome for s in self.sources))

    @property
    def failed(self) -> list[SourceReport]:
        return [s for s in self.sources if s.outcome == "failed"]

    @property
    def probes_failed(self) -> list[ProbeResult]:
        return [p for p in self.probes if not p.found]

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["counts"] = self.counts
        return body


def document_checksum(blocks: tuple[Block, ...], params: ChunkParams) -> str:
    body = "\n".join(f"{'#' * b.level} {b.text}" if b.level else b.text for b in blocks)
    return hashlib.sha256(f"{params.identity}\n{body}".encode()).hexdigest()


def normalized_blocks(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    out = []
    for block in blocks:
        text = normalize_text(block.text)
        if text:
            out.append(Block(block.level, text))
    return tuple(out)


async def _ingest_source(
    entry: SourceEntry, store: KnowledgeStore, params: ChunkParams, min_words: int
) -> SourceReport:
    key = entry.meta.key
    try:
        snapshot = read_snapshot(entry.snapshot)
    except SnapshotError as exc:
        return SourceReport(key, "failed", failure=str(exc))
    if snapshot.key != key or snapshot.url != entry.meta.url:
        return SourceReport(key, "failed", failure="snapshot does not match the manifest entry")
    blocks = normalized_blocks(snapshot.page.blocks)
    words = sum(len(b.text.split()) for b in blocks if b.level == 0)
    if words < min_words:
        return SourceReport(key, "failed", words=words, failure="too little text")
    chunks = chunk_page(entry.meta.title, blocks, params)
    if not chunks:
        return SourceReport(key, "failed", words=words, failure="no chunks")
    result = await store.upsert_document(entry.meta, document_checksum(blocks, params), chunks)
    return SourceReport(
        key,
        result.outcome.value,
        version=result.version,
        words=words,
        chunks=result.chunks_written,
        duplicate_chunks=result.duplicate_chunks,
        duplicate_of=result.duplicate_of,
    )


async def embed_pending(
    store: KnowledgeStore, embeddings: EmbeddingClient
) -> tuple[int, str | None]:
    """Embeds current chunks that have no vector from this model (new chunks,
    or all of them after a model change). Returns (count, failure)."""
    done = 0
    model = embeddings.model_id
    while True:
        pending = await store.chunks_needing_embedding(model, limit=EMBED_BATCH)
        if not pending:
            return done, None
        try:
            vectors = await embeddings.embed(
                [embedding_text(heading, text) for _, heading, text in pending], kind="document"
            )
        except EmbeddingError as exc:
            return done, exc.message
        await store.set_embeddings(
            [(chunk_id, vector) for (chunk_id, _, _), vector in zip(pending, vectors, strict=True)],
            model,
        )
        done += len(pending)


async def run_probes(manifest: Manifest, retriever: KnowledgeRetriever) -> list[ProbeResult]:
    """The ingestion's retrieval checks: each probe query should return its
    own source among the results."""
    results: list[ProbeResult] = []
    for entry in manifest.sources:
        for query in entry.probes:
            outcome = await retriever.retrieve(query, NO_FILTER)
            keys = [p.candidate.source_key for p in outcome.passages]
            rank = keys.index(entry.meta.key) + 1 if entry.meta.key in keys else None
            results.append(ProbeResult(entry.meta.key, query, rank is not None, rank, outcome.mode))
    return results


async def ready(embeddings: EmbeddingClient, *, timeout_seconds: float = 180.0) -> None:
    """Waits for a local model to load (the first run downloads it)."""
    embeddings.start()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            await embeddings.embed(["warm up"], kind="query")
            return
        except EmbeddingError:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.25)


async def ingest(
    manifest: Manifest,
    store: KnowledgeStore,
    *,
    params: ChunkParams,
    embeddings: EmbeddingClient | None,
    retriever: KnowledgeRetriever | None = None,
    prune: bool = False,
    only: set[str] | None = None,
    min_document_words: int = MIN_DOCUMENT_WORDS,
) -> IngestReport:
    started = time.monotonic()
    report = IngestReport(
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        manifest=manifest.path.name,
        chunking=params.identity,
        embedding_model=embeddings.model_id if embeddings is not None else None,
    )
    for entry in manifest.sources:
        if only is not None and entry.meta.key not in only:
            continue
        report.sources.append(await _ingest_source(entry, store, params, min_document_words))
    if prune and only is None:
        report.retired = await store.retire_missing({s.meta.key for s in manifest.sources})
    if embeddings is not None:
        report.embedded, report.embedding_failure = await embed_pending(store, embeddings)
    if retriever is not None:
        report.probes = await run_probes(manifest, retriever)
    report.corpus = await store.stats()
    report.seconds = round(time.monotonic() - started, 2)
    return report


__all__ = ["IngestOutcome", "IngestReport", "ingest"]

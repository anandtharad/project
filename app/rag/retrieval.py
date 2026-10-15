"""Hybrid knowledge retrieval (spec §11.6).

1. Two candidate lists from app/rag/store.py: full-text and vector.
2. Relevance gate. With a query vector, a chunk is a candidate only if its
   cosine similarity reaches `min_semantic`; a lexical match alone is not
   enough (common words such as "sleep" match many chunks). Without a query
   vector (no embedding provider, or a failed call), the lexical fallback
   needs `min_lexical` instead, and the result is marked lexical.
3. Rank merging: reciprocal rank fusion over both lists
   (score = sum of 1 / (rrf_k + rank)), so neither score scale dominates.
4. De-duplication (identical chunk text) and at most `max_per_document`
   chunks per document, then the top `max_results`.

Irrelevant results are rejected rather than forced into the reply (§11.9):
an empty result is a normal outcome. No reranker: add one only if the
evaluation shows it helps (docs/rag.md).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy.exc import SQLAlchemyError

from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.observability import log_event
from app.rag.store import NO_FILTER, Candidate, KnowledgeFilter, KnowledgeStore

logger = logging.getLogger("app.rag")

RetrievalMode = Literal["hybrid", "lexical"]
RetrievalStatus = Literal["ok", "no_results", "unavailable"]


@dataclass(frozen=True)
class RetrievalParams:
    max_results: int = 3
    # Candidates taken from each list before merging.
    pool: int = 20
    # Calibrated for the default local embedding model (docs/rag.md); another
    # embedding model needs its own value.
    min_semantic: float = 0.70
    # ts_rank_cd with normalization 32, used only without a query vector.
    min_lexical: float = 0.10
    max_per_document: int = 2
    rrf_k: int = 60


@dataclass(frozen=True)
class Passage:
    candidate: Candidate
    score: float
    semantic_rank: int | None
    lexical_rank: int | None


@dataclass(frozen=True)
class RetrievalOutcome:
    passages: tuple[Passage, ...]
    status: RetrievalStatus
    mode: RetrievalMode | None
    latency_ms: float
    failure: str | None = None
    # Candidates seen before the relevance gate (evaluation and logs).
    considered: int = 0


def merge(
    lexical: Sequence[Candidate],
    semantic: Sequence[Candidate] | None,
    params: RetrievalParams,
) -> tuple[list[Passage], int]:
    """Gate, fuse, de-duplicate and cap. `semantic` is None when there was no
    query vector (lexical fallback). Shared with the evaluation script."""
    by_id: dict[object, Candidate] = {}
    semantic_rank: dict[object, int] = {}
    lexical_rank: dict[object, int] = {}
    for rank, candidate in enumerate(semantic or (), start=1):
        by_id[candidate.chunk_id] = candidate
        semantic_rank[candidate.chunk_id] = rank
    for rank, candidate in enumerate(lexical, start=1):
        known = by_id.get(candidate.chunk_id)
        by_id[candidate.chunk_id] = (
            replace(known, lexical=candidate.lexical) if known is not None else candidate
        )
        lexical_rank[candidate.chunk_id] = rank

    kept: list[Passage] = []
    for chunk_id, candidate in by_id.items():
        if semantic is not None:
            relevant = candidate.semantic is not None and candidate.semantic >= params.min_semantic
        else:
            relevant = (candidate.lexical or 0.0) >= params.min_lexical
        if not relevant:
            continue
        s_rank, l_rank = semantic_rank.get(chunk_id), lexical_rank.get(chunk_id)
        score = sum(1.0 / (params.rrf_k + r) for r in (s_rank, l_rank) if r is not None)
        kept.append(Passage(candidate, score, s_rank, l_rank))
    kept.sort(key=lambda p: (-p.score, str(p.candidate.chunk_id)))

    selected: list[Passage] = []
    seen_text: set[str] = set()
    per_document: dict[object, int] = {}
    for passage in kept:
        candidate = passage.candidate
        if candidate.text_checksum in seen_text:
            continue
        if per_document.get(candidate.document_id, 0) >= params.max_per_document:
            continue
        seen_text.add(candidate.text_checksum)
        per_document[candidate.document_id] = per_document.get(candidate.document_id, 0) + 1
        selected.append(passage)
        if len(selected) == params.max_results:
            break
    return selected, len(by_id)


class KnowledgeRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingClient | None,
        *,
        params: RetrievalParams,
        embedding_timeout_seconds: float,
    ) -> None:
        self.store = store
        self._embeddings = embeddings
        self.params = params
        self._embedding_timeout = embedding_timeout_seconds

    @property
    def embedding_model(self) -> str | None:
        return self._embeddings.model_id if self._embeddings is not None else None

    async def _query_vector(self, query: str) -> tuple[list[float] | None, str | None]:
        if self._embeddings is None:
            return None, None
        try:
            async with asyncio.timeout(self._embedding_timeout):
                vectors = await self._embeddings.embed([query], kind="query")
            return vectors[0], None
        except EmbeddingError:
            return None, "embedding_failed"
        except TimeoutError:
            return None, "embedding_timeout"
        except Exception:  # a client bug must not fail the turn
            return None, "embedding_error"

    async def retrieve(self, query: str, filters: KnowledgeFilter = NO_FILTER) -> RetrievalOutcome:
        started = time.monotonic()
        vector, failure = await self._query_vector(query)
        model = self.embedding_model
        try:
            lexical = await self.store.lexical(query, limit=self.params.pool, filters=filters)
            semantic = (
                await self.store.semantic(vector, model, limit=self.params.pool, filters=filters)
                if vector is not None and model is not None
                else None
            )
        except (SQLAlchemyError, OSError):
            log_event(logger, "knowledge.store_unavailable", level=logging.WARNING)
            return RetrievalOutcome(
                (), "unavailable", None, _ms(started), failure="store_unavailable"
            )
        passages, considered = merge(lexical, semantic, self.params)
        mode: RetrievalMode = "hybrid" if semantic is not None else "lexical"
        outcome = RetrievalOutcome(
            tuple(passages),
            "ok" if passages else "no_results",
            mode,
            _ms(started),
            failure=failure,
            considered=considered,
        )
        log_event(
            logger,
            "knowledge.retrieved",
            mode=mode,
            count=len(passages),
            considered=considered,
            embedding_failure=failure,
            latency_ms=outcome.latency_ms,
        )
        return outcome


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 2)

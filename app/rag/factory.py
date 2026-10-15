"""Builds the knowledge service from settings (KNOWLEDGE_* in app/config.py)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.embeddings.base import EmbeddingClient
from app.rag.chunking import ChunkParams
from app.rag.retrieval import KnowledgeRetriever, RetrievalParams
from app.rag.service import KnowledgeService
from app.rag.store import KnowledgeStore


def retrieval_params(settings: Settings) -> RetrievalParams:
    return RetrievalParams(
        max_results=settings.knowledge_max_results,
        min_semantic=settings.knowledge_min_semantic,
        min_lexical=settings.knowledge_min_lexical,
        max_per_document=settings.knowledge_max_per_document,
    )


def chunk_params(settings: Settings) -> ChunkParams:
    return ChunkParams(
        max_words=settings.knowledge_chunk_max_words,
        overlap_words=settings.knowledge_chunk_overlap_words,
        min_words=settings.knowledge_chunk_min_words,
    )


def build_retriever(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    embeddings: EmbeddingClient | None,
) -> KnowledgeRetriever:
    return KnowledgeRetriever(
        KnowledgeStore(sessions),
        embeddings,
        params=retrieval_params(settings),
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
    )


def build_knowledge_service(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    embeddings: EmbeddingClient | None,
) -> KnowledgeService | None:
    """None when KNOWLEDGE_ENABLED=false."""
    if not settings.knowledge_enabled:
        return None
    return KnowledgeService(
        build_retriever(settings, sessions, embeddings),
        timeout_seconds=settings.knowledge_timeout_seconds,
    )

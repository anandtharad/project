"""Build the embedding client from settings (SDKs are imported lazily)."""

from __future__ import annotations

from app.config import Settings
from app.embeddings.base import EmbeddingClient, FakeEmbeddingClient
from app.llm.providers.registry import ProviderConfigError

# Defaults per provider (see docs/memory.md for how each was checked).
DEFAULT_MODELS = {
    "local": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}


def build_embedding_client(settings: Settings) -> EmbeddingClient | None:
    provider = settings.embedding_provider
    if provider == "none":
        return None
    if provider == "fake":
        return FakeEmbeddingClient()
    model = settings.embedding_model or DEFAULT_MODELS[provider]
    try:
        if provider == "local":
            from app.embeddings.local import LocalEmbeddingClient

            try:
                return LocalEmbeddingClient(
                    model=model,
                    timeout_seconds=settings.embedding_timeout_seconds,
                    cache_dir=settings.embedding_cache_dir,
                )
            except ValueError as exc:  # fastembed does not know the model
                raise ProviderConfigError(
                    f"EMBEDDING_MODEL={model} is not a fastembed text model"
                ) from exc
        if provider == "openai":
            from app.embeddings.openai_embeddings import OpenAIEmbeddingClient

            if settings.openai_api_key is None:
                raise ProviderConfigError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
            return OpenAIEmbeddingClient(
                api_key=settings.openai_api_key.get_secret_value(),
                model=model,
                timeout_seconds=settings.embedding_timeout_seconds,
                dimensions=settings.embedding_dimensions,
            )
        from app.embeddings.gemini_embeddings import GeminiEmbeddingClient

        if settings.gemini_api_key is None:
            raise ProviderConfigError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY")
        return GeminiEmbeddingClient(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=model,
            timeout_seconds=settings.embedding_timeout_seconds,
            dimensions=settings.embedding_dimensions,
        )
    except ImportError as exc:
        extra = {"local": "embeddings", "openai": "openai", "gemini": "gemini"}[provider]
        raise ProviderConfigError(
            f"EMBEDDING_PROVIDER={provider} needs its package: uv sync --extra {extra}"
        ) from exc

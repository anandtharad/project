"""Embedding provider interface (spec §4.1: configurable embedding provider).

Embeddings power semantic memory retrieval. They are optional: with
EMBEDDING_PROVIDER=none, or when a call fails, retrieval falls back to
PostgreSQL full-text search (spec §23: lexical fallback). Vectors from
different models are never compared: every stored vector records its model.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Literal, Protocol

EmbeddingProviderName = Literal["none", "local", "openai", "gemini", "fake"]
InputKind = Literal["query", "document"]


class EmbeddingError(Exception):
    """An embedding call failed. `message` is safe to log (no content)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmbeddingClient(Protocol):
    @property
    def model_id(self) -> str:
        """Identifier stored next to every vector (provider:model:dimensions)."""
        ...

    async def embed(self, texts: Sequence[str], *, kind: InputKind) -> list[list[float]]:
        """One unit-length vector per text. Raises EmbeddingError."""
        ...

    async def aclose(self) -> None: ...


def normalized(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("embedding has zero or non-finite length")
    return [float(v) / norm for v in vector]


class FakeEmbeddingClient:
    """Deterministic development/test stub: hashed bag of words. Texts sharing
    words get similar vectors; it has no semantic understanding."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"fake:bag-of-words:{self._dimensions}"

    async def embed(self, texts: Sequence[str], *, kind: InputKind) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for word in text.lower().split():
                word = "".join(ch for ch in word if ch.isalnum())
                if word:
                    digest = hashlib.sha256(word.encode()).digest()
                    vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
            if not any(vector):  # no words: a fixed unit vector
                vector[0] = 1.0
            vectors.append(normalized(vector))
        return vectors

    async def aclose(self) -> None:
        return None

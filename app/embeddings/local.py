"""Local embeddings with fastembed: ONNX Runtime on the CPU, no API key.

Facts this relies on (fastembed 0.9.0, read from its source and checked by
running it; see docs/memory.md):
- `TextEmbedding(model_name, cache_dir=..., threads=...)` downloads the model
  on first use (BAAI/bge-small-en-v1.5: about 65 MB from huggingface.co) and
  loads it. Both steps block, and a failed download retries for about 40 s
  before raising.
- `embed()` returns a lazy generator of float32 numpy arrays that are already
  L2-normalized. It has to be consumed in the worker thread, or inference
  would run on the event loop.
- For this model, `query_embed()` and `passage_embed()` are the same as
  `embed()` (no prefix), so queries and documents are embedded the same way.
  Adding a prefix later would need a new `model_id`.
- Input beyond 512 tokens is truncated silently. Memories are at most 200
  characters, so this never applies to them.
- One instance can be shared between threads (onnxruntime's Run() is
  thread-safe).

Loading happens in the background (`start()`), never on the request path:
until the model is ready, `embed()` raises EmbeddingError and retrieval uses
full-text matching. Inference runs on one dedicated worker thread, because
onnxruntime already uses every core for a single call; a call that times out
while still queued is dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.embeddings.base import EmbeddingError, InputKind
from app.observability import log_event

logger = logging.getLogger("app.embeddings")

# After a failed load (no network, say), wait this long before trying again.
RELOAD_AFTER_SECONDS = 300.0
_BATCH_SIZE = 64


class LocalEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        # Imported here so the package stays optional (ImportError -> config hint).
        from fastembed import TextEmbedding

        self._factory: Any = TextEmbedding
        self._model_name = model
        # Model metadata only, no download; ValueError for an unknown model.
        self._dimensions = int(TextEmbedding.get_embedding_size(model))
        self._timeout = timeout_seconds
        self._cache_dir = cache_dir
        self._threads = threads
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddings")
        self._model: Any = None
        self._load: asyncio.Future[Any] | None = None
        self._failed_at: float | None = None

    @property
    def model_id(self) -> str:
        return f"local:{self._model_name}:{self._dimensions}"

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _load_sync(self) -> Any:
        started = time.monotonic()
        model = self._factory(self._model_name, cache_dir=self._cache_dir, threads=self._threads)
        log_event(
            logger,
            "embeddings.model_loaded",
            model=self.model_id,
            seconds=round(time.monotonic() - started, 2),
        )
        return model

    def start(self) -> None:
        """Begin loading (and on first run downloading) the model in the background."""
        if self._model is not None or (self._load is not None and not self._load.done()):
            return
        if (
            self._failed_at is not None
            and time.monotonic() - self._failed_at < RELOAD_AFTER_SECONDS
        ):
            return
        self._load = asyncio.get_running_loop().run_in_executor(self._executor, self._load_sync)
        self._load.add_done_callback(self._loaded)

    def _loaded(self, future: asyncio.Future[Any]) -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is None:
            self._model = future.result()
            self._failed_at = None
            return
        self._failed_at = time.monotonic()
        log_event(
            logger,
            "embeddings.model_load_failed",
            level=logging.WARNING,
            model=self.model_id,
            error_type=type(exc).__name__,
        )

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=_BATCH_SIZE)]

    async def embed(self, texts: Sequence[str], *, kind: InputKind) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self.start()
            raise EmbeddingError("local embedding model is not loaded yet")
        loop = asyncio.get_running_loop()
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._embed_sync, list(texts)), self._timeout
            )
        except TimeoutError as exc:
            raise EmbeddingError("local embedding timed out") from exc
        except Exception as exc:  # onnxruntime/tokenizer errors
            raise EmbeddingError(f"local embedding failed: {type(exc).__name__}") from exc
        if len(vectors) != len(texts) or any(len(v) != self._dimensions for v in vectors):
            raise EmbeddingError("local embedding returned an unexpected shape")
        return vectors

    async def aclose(self) -> None:
        # A running load or inference finishes in its thread; nothing new starts.
        self._executor.shutdown(wait=False, cancel_futures=True)

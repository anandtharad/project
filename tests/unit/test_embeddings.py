"""Embedding clients, offline.

The OpenAI and Gemini clients run through their real SDKs against a mocked
HTTP transport, so these tests pin the request each one sends. The local
client runs against a stand-in for fastembed's TextEmbedding; one test uses
the real model when it is already in the local cache.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterator
from typing import Any

import httpx
import httpx2
import pytest

from app.embeddings.base import EmbeddingError, FakeEmbeddingClient
from app.embeddings.factory import build_embedding_client
from app.embeddings.gemini_embeddings import GeminiEmbeddingClient
from app.embeddings.openai_embeddings import OpenAIEmbeddingClient
from app.llm.providers.registry import ProviderConfigError
from tests.helpers import make_settings


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


# --- fake (development/test stub) ------------------------------------------


async def test_fake_embeddings_are_unit_length_and_share_words() -> None:
    client = FakeEmbeddingClient()
    a, b, c, empty = await client.embed(
        ["my dog Biscuit", "Biscuit the dog", "tax returns", ""], kind="document"
    )
    assert all(abs(norm(v) - 1) < 1e-9 for v in (a, b, c, empty))
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    assert dot > sum(x * y for x, y in zip(a, c, strict=True))


# --- OpenAI -----------------------------------------------------------------


def openai_client(handler: Any, **options: Any) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        api_key="sk-test",
        model="text-embedding-3-small",
        timeout_seconds=5,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **options,
    )


async def test_openai_request_shape_and_index_order() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith("/embeddings")
        sent.append(json.loads(request.content))
        # Out of order on purpose: the client must sort by index.
        data = [
            {"object": "embedding", "index": 1, "embedding": [0.0, 2.0]},
            {"object": "embedding", "index": 0, "embedding": [3.0, 4.0]},
        ]
        body = {
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
        return httpx2.Response(200, json=body)

    client = openai_client(handler, dimensions=2)
    vectors = await client.embed(["first", "second"], kind="query")
    assert sent == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
            "dimensions": 2,
        }
    ]
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]  # normalized, in input order
    assert client.model_id == "openai:text-embedding-3-small:2"
    await client.aclose()


async def test_openai_omits_dimensions_by_default_and_maps_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert "dimensions" not in json.loads(request.content)
        return httpx2.Response(429, json={"error": {"message": "slow down", "type": "rate"}})

    client = openai_client(handler)
    with pytest.raises(EmbeddingError) as info:
        await client.embed(["x"], kind="document")
    assert info.value.message == "openai embeddings: HTTP 429"
    assert await client.embed([], kind="query") == []
    await client.aclose()


# --- Gemini -----------------------------------------------------------------


def gemini_client(handler: Any, **options: Any) -> GeminiEmbeddingClient:
    return GeminiEmbeddingClient(
        api_key="g-test",
        model="gemini-embedding-001",
        timeout_seconds=5,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **options,
    )


async def test_gemini_batch_request_uses_retrieval_task_types() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/gemini-embedding-001:batchEmbedContents")
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(
            200, json={"embeddings": [{"values": [1.0, 1.0]} for _ in body["requests"]]}
        )

    client = gemini_client(handler, dimensions=2)
    vectors = await client.embed(["a", "b"], kind="query")
    await client.embed(["c"], kind="document")
    assert [r["taskType"] for r in sent[0]["requests"]] == ["RETRIEVAL_QUERY"] * 2
    assert sent[0]["requests"][0]["outputDimensionality"] == 2
    assert sent[0]["requests"][0]["content"]["parts"] == [{"text": "a"}]
    assert sent[1]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert all(abs(norm(v) - 1) < 1e-9 for v in vectors)
    await client.aclose()


async def test_gemini_errors_and_short_responses_raise() -> None:
    client = gemini_client(lambda r: httpx.Response(400, json={"error": {"code": 400}}))
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        await client.embed(["a"], kind="query")
    client = gemini_client(lambda r: httpx.Response(200, json={"embeddings": []}))
    with pytest.raises(EmbeddingError, match="unexpected number"):
        await client.embed(["a"], kind="query")


# --- local (fastembed) --------------------------------------------------------


class StubTextEmbedding:
    """Stands in for fastembed.TextEmbedding: same call shapes, no model."""

    load_gate: threading.Event | None = None
    fail_load = False
    loads = 0

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: Any = None):
        type(self).loads += 1
        if self.load_gate is not None:
            self.load_gate.wait(5)
        if self.fail_load:
            raise ValueError("Could not load model from any source.")
        self.cache_dir = cache_dir

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        if model_name != "BAAI/bge-small-en-v1.5":
            raise ValueError("not supported")
        return 3

    def embed(self, documents: list[str], batch_size: int = 256) -> Iterator[Any]:
        import numpy as np

        for text in documents:
            yield np.array([len(text), 1.0, 0.0], dtype=np.float32) / np.float32(
                math.sqrt(len(text) ** 2 + 1)
            )


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[StubTextEmbedding]:
    import fastembed

    StubTextEmbedding.load_gate, StubTextEmbedding.fail_load, StubTextEmbedding.loads = (
        None,
        False,
        0,
    )
    monkeypatch.setattr(fastembed, "TextEmbedding", StubTextEmbedding)
    return StubTextEmbedding


def local_client(**options: Any) -> Any:
    from app.embeddings.local import LocalEmbeddingClient

    return LocalEmbeddingClient(model="BAAI/bge-small-en-v1.5", timeout_seconds=2, **options)


async def wait_until(condition: Any, seconds: float = 2.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def test_local_client_loads_in_the_background_and_embeds(
    stub_fastembed: type[StubTextEmbedding],
) -> None:
    client = local_client(cache_dir="models")
    assert client.model_id == "local:BAAI/bge-small-en-v1.5:3"
    stub_fastembed.load_gate = threading.Event()
    client.start()
    # Still loading: the caller is told at once (retrieval falls back to full text).
    with pytest.raises(EmbeddingError, match="not loaded yet"):
        await client.embed(["hello"], kind="query")
    stub_fastembed.load_gate.set()
    await wait_until(lambda: client.ready)
    vectors = await client.embed(["hi", "hello"], kind="document")
    assert len(vectors) == 2 and all(abs(norm(v) - 1) < 1e-6 for v in vectors)
    assert stub_fastembed.loads == 1  # start() and embed() did not load twice
    assert await client.embed([], kind="query") == []
    await client.aclose()


async def test_a_failed_load_is_not_retried_immediately(
    stub_fastembed: type[StubTextEmbedding],
) -> None:
    stub_fastembed.fail_load = True
    client = local_client()
    client.start()
    await wait_until(lambda: client._failed_at is not None)
    with pytest.raises(EmbeddingError, match="not loaded yet"):
        await client.embed(["hello"], kind="query")
    await asyncio.sleep(0.05)
    assert stub_fastembed.loads == 1  # embed() did not start a second ~40 s attempt
    await client.aclose()


async def test_local_inference_errors_and_timeouts_become_embedding_errors(
    stub_fastembed: type[StubTextEmbedding], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = local_client()
    client.start()
    await wait_until(lambda: client.ready)

    def broken(texts: list[str], batch_size: int = 64) -> Any:
        raise RuntimeError("onnx")

    monkeypatch.setattr(client._model, "embed", broken)
    with pytest.raises(EmbeddingError, match="RuntimeError"):
        await client.embed(["x"], kind="query")

    release = threading.Event()

    def slow(texts: list[str], batch_size: int = 64) -> Any:
        release.wait(5)
        return iter([])

    monkeypatch.setattr(client._model, "embed", slow)
    client._timeout = 0.05
    with pytest.raises(EmbeddingError, match="timed out"):
        await client.embed(["x"], kind="query")
    release.set()
    await client.aclose()


def test_factory_builds_each_provider_or_explains_what_is_missing(
    stub_fastembed: type[StubTextEmbedding],
) -> None:
    assert build_embedding_client(make_settings(embedding_provider="none")) is None
    assert isinstance(build_embedding_client(make_settings()), FakeEmbeddingClient)
    local = build_embedding_client(make_settings(embedding_provider="local"))
    assert local is not None and local.model_id == "local:BAAI/bge-small-en-v1.5:3"
    with pytest.raises(ProviderConfigError, match="not a fastembed text model"):
        build_embedding_client(make_settings(embedding_provider="local", embedding_model="nope"))
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_embedding_client(make_settings(embedding_provider="openai"))
    with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY"):
        build_embedding_client(make_settings(embedding_provider="gemini"))


def cached_model_dir() -> str | None:
    # fastembed's own default: FASTEMBED_CACHE_PATH, else <temp>/fastembed_cache.
    default = os.path.join(tempfile.gettempdir(), "fastembed_cache")
    for path in (os.environ.get("FASTEMBED_CACHE_PATH"), default):
        if path and os.path.isdir(os.path.join(path, "models--Qdrant--bge-small-en-v1.5-onnx-Q")):
            return path
    return None


@pytest.mark.skipif(
    cached_model_dir() is None, reason="bge-small-en-v1.5 is not in the local cache"
)
async def test_real_local_model_ranks_the_related_memory_first() -> None:
    from app.embeddings.local import LocalEmbeddingClient

    client = LocalEmbeddingClient(
        model="BAAI/bge-small-en-v1.5", timeout_seconds=30, cache_dir=cached_model_dir()
    )
    client.start()
    await wait_until(lambda: client.ready or client._failed_at is not None, seconds=60)
    assert client.ready
    query, *docs = await client.embed(
        ["What's my dog called?", "has a rescue greyhound called Biscuit", "studies history"],
        kind="query",
    )
    scores = [sum(q * d for q, d in zip(query, doc, strict=True)) for doc in docs]
    assert scores[0] > scores[1]
    assert all(abs(norm(v) - 1) < 1e-4 for v in docs)
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, text="<html>captive portal</html>"),
        httpx2.Response(200, json={"object": "list", "data": "nope"}),
        httpx2.Response(200, text="{not json"),
    ],
)
async def test_openai_malformed_bodies_become_embedding_errors(response: httpx2.Response) -> None:
    client = openai_client(lambda r: response)
    with pytest.raises(EmbeddingError):
        await client.embed(["x"], kind="query")


async def test_openai_embeddings_are_not_retried_inside_a_turn() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(429, headers={"retry-after": "30"}, json={"error": {}})

    client = OpenAIEmbeddingClient(
        api_key="sk-test",
        model="text-embedding-3-small",
        timeout_seconds=5,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(EmbeddingError, match="429"):
        await client.embed(["x"], kind="query")
    assert calls == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>captive portal</html>"),
        httpx.Response(200, json={"embeddings": [{"values": "nope"}]}),
    ],
)
async def test_gemini_malformed_bodies_become_embedding_errors(response: httpx.Response) -> None:
    client = gemini_client(lambda r: response)
    with pytest.raises(EmbeddingError):
        await client.embed(["x"], kind="query")


async def test_a_finished_load_is_used_rather_than_started_again(
    stub_fastembed: type[StubTextEmbedding],
) -> None:
    client = local_client()
    client.start()
    await asyncio.wait({client._load})
    # The load finished but its done-callback may not have run yet.
    client._model = None
    vectors = await client.embed(["hi"], kind="query")
    assert len(vectors) == 1 and stub_fastembed.loads == 1
    await client.aclose()

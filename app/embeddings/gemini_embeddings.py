"""Gemini embeddings through the official google-genai SDK.

SDK contract (google-genai 2.29, read from the installed package):
`client.aio.models.embed_content(model=..., contents=[...],
config=EmbedContentConfig(task_type=..., output_dimensionality=...))`
returns `embeddings`, a list of `ContentEmbedding(values)`, one per input.
Queries use the RETRIEVAL_QUERY task type and memories RETRIEVAL_DOCUMENT.
Vectors are normalized here, since a reduced output dimensionality may not
come back unit-length.

Not live-tested in this project (no Gemini key was available); the offline
tests run it against a mocked HTTP transport.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from google import genai
from google.genai import errors, types

from app.embeddings.base import EmbeddingError, InputKind, normalized

_TASK_TYPES: dict[InputKind, str] = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}


class GeminiEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        dimensions: int | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Owning the httpx client pins the transport, as in the Gemini LLM adapter.
        self._http = http_client or httpx.AsyncClient()
        self._owns_http = http_client is None
        self._client = genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                base_url=base_url,
                httpx_async_client=self._http,
                timeout=int(timeout_seconds * 1000),
            ),
        )
        self._model = model
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"gemini:{self._model}:{self._dimensions or 'default'}"

    async def embed(self, texts: Sequence[str], *, kind: InputKind) -> list[list[float]]:
        if not texts:
            return []
        contents: list[types.ContentUnion] = list(texts)
        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=_TASK_TYPES[kind], output_dimensionality=self._dimensions
                ),
            )
        except errors.APIError as exc:
            raise EmbeddingError(f"gemini embeddings: HTTP {exc.code}") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"gemini embeddings: {type(exc).__name__}") from exc
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts) or any(not e.values for e in embeddings):
            raise EmbeddingError("gemini embeddings: unexpected number of vectors")
        return [normalized(e.values or []) for e in embeddings]

    def start(self) -> None:
        return None

    async def aclose(self) -> None:
        await self._client.aio.aclose()
        if self._owns_http:
            await self._http.aclose()

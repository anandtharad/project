"""OpenAI embeddings (`POST /v1/embeddings` through the official SDK).

SDK contract (openai 3.29 with its httpx2 transport, read from the installed package):
`AsyncOpenAI.embeddings.create(input=[...], model=..., dimensions=...,
encoding_format="float")` returns `data`, a list of `Embedding(embedding,
index)`. Results are re-ordered by `index` rather than assuming the order.

Not live-tested in this project (no OpenAI key was available); the offline
tests run it against a mocked HTTP transport.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx2
import openai

from app.embeddings.base import EmbeddingError, InputKind, normalized


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        dimensions: int | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout_seconds,
            http_client=http_client,
        )
        self._model = model
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}:{self._dimensions or 'default'}"

    async def embed(self, texts: Sequence[str], *, kind: InputKind) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                input=list(texts),
                model=self._model,
                encoding_format="float",
                dimensions=self._dimensions if self._dimensions else openai.omit,
            )
        except openai.APIStatusError as exc:
            raise EmbeddingError(f"openai embeddings: HTTP {exc.status_code}") from exc
        except openai.APIError as exc:
            raise EmbeddingError(f"openai embeddings: {type(exc).__name__}") from exc
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise EmbeddingError("openai embeddings: unexpected number of vectors")
        return [normalized(item.embedding) for item in data]

    def start(self) -> None:
        return None

    async def aclose(self) -> None:
        await self._client.close()

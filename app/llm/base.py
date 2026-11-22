from __future__ import annotations

from typing import Protocol

from app.llm.schemas import GenerationRequest, GenerationResponse


class LLMClient(Protocol):
    """Provider-neutral generation interface.

    Implementations raise `LLMError` for every provider failure and return a
    `GenerationResponse` otherwise; a response may still be unusable (refusal,
    truncation), which the response validator decides.

    Structured output: when `request.response_schema` is set, the reply
    content is JSON text intended to match that schema (see `ResponseSchema`)
    and `GenerationResponse.structured_mode` says how it was obtained.
    """

    @property
    def provider(self) -> str: ...

    # Adapters that implement tool calling set `supports_tools = True`
    # (Groq and the test doubles); read it with getattr(client,
    # "supports_tools", False). Requests with tools go only to those.

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    async def aclose(self) -> None: ...

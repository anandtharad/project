from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.providers.anthropic_provider import AnthropicLLMClient
from app.llm.providers.fake import FakeLLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.generation_provider == "fake":
        return FakeLLMClient()
    api_key = (
        settings.generation_api_key.get_secret_value() if settings.generation_api_key else None
    )
    return AnthropicLLMClient(
        api_key=api_key,
        max_retries=settings.generation_max_retries,
        refusal_fallback=settings.generation_refusal_fallback,
    )

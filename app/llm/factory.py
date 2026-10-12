"""Build the configured LLM client.

Provider SDKs are optional extras and are imported only for the selected
provider, so a missing SDK produces a clear configuration error instead of
an import failure at startup.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.providers.registry import AdapterKind, ProviderConfigError, ResolvedProvider


def _missing_sdk(resolved: ResolvedProvider, exc: ImportError) -> ProviderConfigError:
    return ProviderConfigError(
        f"Provider {resolved.preset.name!r} needs its SDK: uv sync --extra {resolved.preset.extra}"
        f" (missing module: {exc.name})"
    )


def build_llm_client(settings: Settings, resolved: ResolvedProvider) -> LLMClient:
    kind = resolved.preset.kind
    if kind is AdapterKind.FAKE:
        from app.llm.providers.fake import FakeLLMClient

        return FakeLLMClient()
    if kind is AdapterKind.ANTHROPIC:
        try:
            from app.llm.providers.anthropic_provider import AnthropicLLMClient
        except ImportError as exc:
            raise _missing_sdk(resolved, exc) from exc
        return AnthropicLLMClient(
            api_key=resolved.api_key,
            max_retries=settings.generation_max_retries,
            refusal_fallback=settings.generation_refusal_fallback,
        )
    raise ProviderConfigError(
        f"Provider {resolved.preset.name!r} ({kind.value} adapter) is not available yet"
    )

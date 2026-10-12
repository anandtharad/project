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
    if kind is AdapterKind.GROQ:
        try:
            from app.llm.providers.groq_provider import GroqLLMClient
        except ImportError as exc:
            raise _missing_sdk(resolved, exc) from exc
        return GroqLLMClient(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            max_retries=settings.generation_max_retries,
        )
    if kind is AdapterKind.GEMINI:
        try:
            from app.llm.providers.gemini_provider import GeminiLLMClient
        except ImportError as exc:
            raise _missing_sdk(resolved, exc) from exc
        if not resolved.api_key:  # guaranteed by resolve_provider; keeps typing honest
            raise ProviderConfigError("Provider 'gemini' needs GEMINI_API_KEY")
        return GeminiLLMClient(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            max_retries=settings.generation_max_retries,
        )
    if kind is AdapterKind.OPENAI_COMPATIBLE:
        try:
            from app.llm.providers.openai_provider import OpenAICompatibleLLMClient
        except ImportError as exc:
            raise _missing_sdk(resolved, exc) from exc
        preset = resolved.preset
        return OpenAICompatibleLLMClient(
            provider=preset.name,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            max_retries=settings.generation_max_retries,
            system_role=preset.system_role,
            max_tokens_param=preset.max_tokens_param,
            effort_style=(
                preset.effort_param
                if preset.effort_param in ("reasoning_effort", "enable_thinking")
                else "none"
            ),
            send_openai_headers=preset.send_openai_headers,
        )
    raise ProviderConfigError(
        f"Provider {resolved.preset.name!r} ({kind.value} adapter) is not available yet"
    )

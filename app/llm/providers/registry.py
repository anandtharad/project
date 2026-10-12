"""Provider presets: how each supported provider is reached.

A preset names the adapter kind, endpoint, API-key setting, and a default
model. GENERATION_MODEL / GENERATION_BASE_URL / GENERATION_API_KEY override
the preset. Default models are starting points only; model choice must come
from measured evaluation (spec §5.2).

Adapter kinds:
- anthropic: official `anthropic` SDK (Messages API)
- groq: official `groq` SDK
- gemini: official `google-genai` SDK
- openai_compatible: official `openai` SDK; used for OpenAI itself and for
  providers whose official docs state OpenAI SDK compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.config import Effort, ProviderName, Settings


class AdapterKind(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    GROQ = "groq"
    GEMINI = "gemini"
    FAKE = "fake"


@dataclass(frozen=True)
class ProviderPreset:
    name: ProviderName
    kind: AdapterKind
    # pip extra providing the SDK (uv sync --extra <extra>)
    extra: str | None
    default_model: str | None
    base_url: str | None
    # Settings attribute holding this provider's key (None: no key needed).
    api_key_setting: str | None
    docs_url: str
    default_effort: Effort | None = None
    # Whether a key is mandatory (local servers such as Ollama need none).
    requires_api_key: bool = True


PRESETS: dict[str, ProviderPreset] = {
    p.name: p
    for p in (
        ProviderPreset(
            name="anthropic",
            kind=AdapterKind.ANTHROPIC,
            extra="anthropic",
            default_model="claude-opus-5-5",
            base_url=None,
            api_key_setting="anthropic_api_key",
            docs_url="https://platform.claude.com/docs",
            default_effort="low",
            # The SDK can also resolve ANTHROPIC_AUTH_TOKEN or a CLI profile.
            requires_api_key=False,
        ),
        ProviderPreset(
            name="openai",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            api_key_setting="openai_api_key",
            docs_url="https://platform.openai.com/docs/api-reference/chat",
        ),
        ProviderPreset(
            name="groq",
            kind=AdapterKind.GROQ,
            extra="groq",
            default_model="llama-3.3-70b-versatile",
            base_url=None,
            api_key_setting="groq_api_key",
            docs_url="https://console.groq.com/docs/models",
        ),
        ProviderPreset(
            name="gemini",
            kind=AdapterKind.GEMINI,
            extra="gemini",
            default_model="gemini-2.5-flash",
            base_url=None,
            api_key_setting="gemini_api_key",
            docs_url="https://ai.google.dev/gemini-api/docs/text-generation",
        ),
        ProviderPreset(
            name="xai",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model="grok-4",
            base_url="https://api.x.ai/v1",
            api_key_setting="xai_api_key",
            docs_url="https://docs.x.ai/docs/guides/chat",
        ),
        ProviderPreset(
            name="qwen",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model="qwen-plus",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key_setting="dashscope_api_key",
            docs_url="https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope",
        ),
        ProviderPreset(
            name="deepseek",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key_setting="deepseek_api_key",
            docs_url="https://api-docs.deepseek.com/",
        ),
        ProviderPreset(
            name="openrouter",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,
            base_url="https://openrouter.ai/api/v1",
            api_key_setting="openrouter_api_key",
            docs_url="https://openrouter.ai/docs/quickstart",
        ),
        ProviderPreset(
            name="together",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,
            base_url="https://api.together.xyz/v1",
            api_key_setting="together_api_key",
            docs_url="https://docs.together.ai/docs/openai-api-compatibility",
        ),
        ProviderPreset(
            name="ollama",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,
            base_url="http://localhost:11434/v1",
            api_key_setting=None,
            docs_url="https://docs.ollama.com/api/openai-compatibility",
            requires_api_key=False,
        ),
        ProviderPreset(
            name="openai_compatible",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,
            base_url=None,
            api_key_setting=None,
            docs_url="https://platform.openai.com/docs/api-reference/chat",
            requires_api_key=False,
        ),
        ProviderPreset(
            name="fake",
            kind=AdapterKind.FAKE,
            extra=None,
            default_model="fake-stub",
            base_url=None,
            api_key_setting=None,
            docs_url="",
            requires_api_key=False,
        ),
    )
}


class ProviderConfigError(ValueError):
    """The selected provider cannot be configured (missing model, key, or SDK)."""


@dataclass(frozen=True)
class ResolvedProvider:
    preset: ProviderPreset
    model: str
    base_url: str | None
    api_key: str | None = None
    effort: Effort | None = None

    def __repr__(self) -> str:  # never expose the key
        return (
            f"ResolvedProvider(name={self.preset.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={'set' if self.api_key else 'unset'})"
        )


def resolve_provider(settings: Settings) -> ResolvedProvider:
    preset = PRESETS[settings.generation_provider]
    model = settings.generation_model or preset.default_model
    if not model:
        raise ProviderConfigError(
            f"GENERATION_MODEL is required for provider {preset.name!r} (no default model)"
        )
    api_key: str | None = None
    if settings.generation_api_key is not None:
        api_key = settings.generation_api_key.get_secret_value()
    elif preset.api_key_setting is not None:
        secret = getattr(settings, preset.api_key_setting)
        api_key = secret.get_secret_value() if secret is not None else None
    if preset.requires_api_key and not api_key:
        key_var = (preset.api_key_setting or "generation_api_key").upper()
        raise ProviderConfigError(
            f"Provider {preset.name!r} needs an API key: set {key_var} or GENERATION_API_KEY"
        )
    return ResolvedProvider(
        preset=preset,
        model=model,
        base_url=settings.generation_base_url or preset.base_url,
        api_key=api_key,
        effort=settings.generation_effort or preset.default_effort,
    )

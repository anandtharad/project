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
from typing import Literal

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
    # Defaults for MAX_OUTPUT_TOKENS / MAX_CONTEXT_TOKENS when those are unset.
    # Output tokens include hidden reasoning tokens on reasoning models.
    default_max_output_tokens: int = 4096
    default_max_context_tokens: int = 16_000
    # --- openai_compatible request shape ---
    # OpenAI wants the instruction as "developer"; compatible servers use "system".
    system_role: Literal["system", "developer"] = "system"
    max_tokens_param: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    # How GENERATION_EFFORT is expressed: "reasoning_effort" (OpenAI-style
    # field), "enable_thinking" (DashScope: enable_thinking + thinking_budget),
    # or "unsupported" (ignored with a startup warning). Native adapters
    # (anthropic, groq, gemini) map effort themselves: "adapter".
    effort_param: Literal["adapter", "reasoning_effort", "enable_thinking", "unsupported"] = (
        "adapter"
    )
    # Effort values the endpoint accepts (reasoning_effort presets only);
    # others are clamped to the nearest accepted value.
    effort_values: tuple[Effort, ...] = ("none", "low", "medium", "high", "xhigh", "max")
    # OpenAI-only headers (organization, project, OPENAI_CUSTOM_HEADERS) are
    # stripped for every other endpoint so they never leak to third parties.
    send_openai_headers: bool = False


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
            # Thinking counts toward output tokens on current Claude models.
            default_max_output_tokens=16_000,
            default_max_context_tokens=32_000,
        ),
        ProviderPreset(
            name="openai",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model="gpt-6-luna",
            # Explicit, so OPENAI_BASE_URL in the environment cannot redirect it.
            base_url="https://api.openai.com/v1",
            api_key_setting="openai_api_key",
            docs_url="https://platform.openai.com/docs/api-reference/chat",
            default_effort="low",
            default_max_output_tokens=8192,
            system_role="developer",
            effort_param="reasoning_effort",
            send_openai_headers=True,
        ),
        ProviderPreset(
            name="groq",
            kind=AdapterKind.GROQ,
            extra="groq",
            # Production model; Groq's named replacement for the retired Llama,
            # Qwen3-32B and Kimi models. Alternatives on the free tier:
            # openai/gpt-oss-20b (faster) and qwen/qwen3.8-27b (preview).
            default_model="openai/gpt-oss-120b",
            base_url=None,
            api_key_setting="groq_api_key",
            docs_url="https://console.groq.com/docs/models",
            default_effort="low",
            # Sized for the free tier (about 8K tokens/minute per model):
            # a turn must fit comfortably inside one minute's budget.
            default_max_output_tokens=2048,
            default_max_context_tokens=4000,
        ),
        ProviderPreset(
            name="gemini",
            kind=AdapterKind.GEMINI,
            extra="gemini",
            # Current Flash model reported on the free tier; gemini-2.5-* is
            # legacy. Alternatives: gemini-3.8-flash, gemini-3.1-flash-lite.
            default_model="gemini-3.6-flash",
            base_url=None,
            api_key_setting="gemini_api_key",
            docs_url="https://ai.google.dev/gemini-api/docs/text-generation",
            default_effort="low",
            # Thinking tokens may count toward the output limit.
            default_max_output_tokens=8192,
        ),
        ProviderPreset(
            name="xai",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            # Cheapest current Grok; accepts reasoning_effort "none".
            # Retired Grok slugs are served by it. Flagship: grok-4.7.
            default_model="grok-4.3",
            base_url="https://api.x.ai/v1",
            api_key_setting="xai_api_key",
            docs_url="https://docs.x.ai/docs/guides/chat",
            effort_param="reasoning_effort",
            # xAI rejects OpenAI-only values ("minimal", "max"); "xhigh" only on
            # some models, so it is clamped to "high".
            effort_values=("none", "low", "medium", "high"),
        ),
        ProviderPreset(
            name="qwen",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            # Hybrid model with thinking OFF by default, so non-streaming
            # calls are safe. (qwen3.5+ models think by default, and the
            # open-source Qwen3 models only stream while thinking.)
            default_model="qwen-plus",
            # Singapore (international) endpoint; keys are region-bound, so
            # override GENERATION_BASE_URL for Beijing/US workspaces.
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key_setting="dashscope_api_key",
            docs_url="https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope",
            # Covers reasoning + answer; DashScope deprecated max_tokens.
            max_tokens_param="max_completion_tokens",
            effort_param="enable_thinking",
        ),
        ProviderPreset(
            name="deepseek",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            # DeepSeek renamed its models twice in 2026 (deepseek-chat was
            # discontinued); set GENERATION_MODEL explicitly.
            default_model=None,
            base_url="https://api.deepseek.com",
            api_key_setting="deepseek_api_key",
            docs_url="https://api-docs.deepseek.com/",
            max_tokens_param="max_tokens",
            effort_param="reasoning_effort",
            effort_values=("none", "low", "high", "max"),
        ),
        ProviderPreset(
            name="openrouter",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,  # vendor/model ids, e.g. openai/gpt-oss-120b
            base_url="https://openrouter.ai/api/v1",
            api_key_setting="openrouter_api_key",
            docs_url="https://openrouter.ai/docs/quickstart",
            max_tokens_param="max_tokens",
            # OpenRouter's reasoning control is its own `reasoning` object.
            effort_param="unsupported",
        ),
        ProviderPreset(
            name="together",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,  # org/model ids, e.g. openai/gpt-oss-120b
            base_url="https://api.together.ai/v1",
            api_key_setting="together_api_key",
            docs_url="https://docs.together.ai/docs/openai-api-compatibility",
            max_tokens_param="max_tokens",
            effort_param="unsupported",
        ),
        ProviderPreset(
            name="mistral",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            # Mistral documents the OpenAI SDK path (migration guide); its
            # request schema rejects unknown fields, and this adapter only
            # sends model, messages, max_tokens, temperature, reasoning_effort.
            default_model="mistral-large-latest",
            base_url="https://api.mistral.ai/v1",
            api_key_setting="mistral_api_key",
            docs_url="https://docs.mistral.ai/",
            max_tokens_param="max_tokens",
            effort_param="reasoning_effort",
            effort_values=("none", "high"),
        ),
        ProviderPreset(
            name="ollama",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            default_model=None,  # whatever model you have pulled
            base_url="http://localhost:11434/v1",
            api_key_setting=None,
            docs_url="https://docs.ollama.com/api/openai-compatibility",
            requires_api_key=False,
            # Ollama's OpenAI layer has no max_completion_tokens.
            max_tokens_param="max_tokens",
            effort_param="reasoning_effort",
            effort_values=("none", "low", "medium", "high", "max"),
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
            # The most widely accepted field across compatible servers.
            max_tokens_param="max_tokens",
            effort_param="unsupported",
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
    max_output_tokens: int
    max_context_tokens: int
    api_key: str | None = None
    effort: Effort | None = None
    warnings: tuple[str, ...] = ()

    def __repr__(self) -> str:  # never expose the key
        return (
            f"ResolvedProvider(name={self.preset.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={'set' if self.api_key else 'unset'})"
        )


_EFFORT_ORDER: tuple[Effort, ...] = ("none", "low", "medium", "high", "xhigh", "max")


def clamp_effort(effort: Effort, allowed: tuple[Effort, ...]) -> Effort:
    """The accepted value closest to `effort`, preferring the lower one on ties."""
    if effort in allowed:
        return effort
    target = _EFFORT_ORDER.index(effort)
    return min(
        allowed, key=lambda e: (abs(_EFFORT_ORDER.index(e) - target), _EFFORT_ORDER.index(e))
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
    effort = settings.generation_effort or preset.default_effort
    warnings: list[str] = []
    if effort is not None and preset.effort_param == "unsupported":
        warnings.append(f"GENERATION_EFFORT is not supported for provider {preset.name!r}; ignored")
        effort = None
    elif effort is not None and preset.effort_param == "reasoning_effort":
        clamped = clamp_effort(effort, preset.effort_values)
        if clamped != effort:
            warnings.append(
                f"GENERATION_EFFORT={effort!r} is not accepted by {preset.name!r}; "
                f"using {clamped!r}"
            )
        effort = clamped
    return ResolvedProvider(
        preset=preset,
        model=model,
        base_url=settings.generation_base_url or preset.base_url,
        max_output_tokens=settings.max_output_tokens or preset.default_max_output_tokens,
        max_context_tokens=settings.max_context_tokens or preset.default_max_context_tokens,
        api_key=api_key,
        effort=effort,
        warnings=tuple(warnings),
    )

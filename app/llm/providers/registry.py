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

Structured output (JSON Schema replies) is a preset field for
openai_compatible presets, resolved per model; native adapters choose per
model themselves. docs/providers.md records each choice and its source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from app.config import Effort, ProviderName, Settings
from app.llm.structured import ChatStructuredOutput

# How a response_schema is requested: an openai_compatible style (see
# app/llm/structured.py), or "adapter" for native adapters, which decide
# per model themselves (documented in docs/providers.md).
StructuredOutput = Literal["adapter"] | ChatStructuredOutput


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
    # Model-name prefixes the preset's default_effort applies to (models whose
    # reasoning controls are known). Empty = only the preset's default model.
    default_effort_models: tuple[str, ...] = ()
    # OpenAI-only headers (organization, project, OPENAI_CUSTOM_HEADERS) are
    # stripped for every other endpoint so they never leak to third parties.
    send_openai_headers: bool = False
    # Structured output support from the provider's docs; per-model
    # exceptions live in _MODEL_STRUCTURED_OUTPUT.
    structured_output: StructuredOutput = "adapter"
    # OpenRouter ignores parameters an upstream endpoint lacks unless
    # provider.require_parameters is set; structured requests set it.
    structured_require_parameters: bool = False


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
            # Structured Outputs (strict json_schema); older models: overrides.
            structured_output="json_schema",
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
            default_effort_models=("openai/gpt-oss", "qwen/"),
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
            # The adapter only sends thinking controls to Gemini 2.5 / 3.x.
            default_effort_models=("gemini-",),
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
            # Documented as guaranteed to match supported schema features.
            structured_output="json_schema",
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
            # JSON Schema mode exists only for some qwen3.7/3.8 models and not
            # in the Singapore region; JSON Object mode covers most models.
            structured_output="json_object",
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
            # response_format accepts only "text" and "json_object".
            structured_output="json_object",
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
            # Enforcement varies by upstream endpoint ("not guaranteed on every
            # endpoint"); require_parameters keeps it from being ignored.
            structured_output="json_schema_best_effort",
            structured_require_parameters=True,
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
            # Supported models only; the docs say to also put the schema in
            # the prompt.
            structured_output="json_schema_best_effort",
        ),
        ProviderPreset(
            name="mistral",
            kind=AdapterKind.OPENAI_COMPATIBLE,
            extra="openai",
            # Mistral documents the OpenAI SDK path (migration guide); its
            # request schema rejects unknown fields, and this adapter only
            # sends model, messages, max_tokens, temperature, reasoning_effort
            # and (structured requests only) response_format.
            default_model="mistral-large-latest",
            base_url="https://api.mistral.ai/v1",
            api_key_setting="mistral_api_key",
            docs_url="https://docs.mistral.ai/",
            max_tokens_param="max_tokens",
            effort_param="reasoning_effort",
            effort_values=("none", "high"),
            # JSON schema mode "guarantees" the schema (API reference).
            structured_output="json_schema",
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
            # Mapped to Ollama's `format`; the docs advise repeating the schema
            # in the prompt, and Ollama's cloud does not support it.
            structured_output="json_schema_best_effort",
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
            # Unknown server: describe the schema in the instruction only.
            structured_output="prompt",
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
    structured_output: StructuredOutput = "adapter"

    def __repr__(self) -> str:  # never expose the key
        return (
            f"ResolvedProvider(name={self.preset.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={'set' if self.api_key else 'unset'})"
        )


_EFFORT_ORDER: tuple[Effort, ...] = ("none", "low", "medium", "high", "xhigh", "max")

# Models documented to reject some effort values their provider otherwise
# accepts (matched by name prefix): reasoning cannot be turned off on these.
_MODEL_EFFORT_VALUES: dict[str, tuple[Effort, ...]] = {
    "grok-4.7": ("low", "medium", "high"),
    "gpt-6-astra": ("low", "medium", "high", "xhigh", "max"),
    "gpt-6.1-sol": ("low", "medium", "high", "xhigh", "max"),
}


# Models documented to lack their preset's structured output mode (matched
# by name prefix, first match wins).
_MODEL_STRUCTURED_OUTPUT: dict[str, tuple[tuple[str, ChatStructuredOutput], ...]] = {
    # json_schema needs gpt-4o-mini / gpt-4o-2024-08-06 or later; JSON mode
    # covers gpt-3.5-turbo and gpt-4-*.
    # A trailing "$" means an exact match ("gpt-4" must not catch gpt-4o).
    "openai": (
        ("gpt-3.5-turbo", "json_object"),
        ("gpt-4-turbo", "json_object"),
        ("gpt-4-1106", "json_object"),
        ("gpt-4-0125", "json_object"),
        ("gpt-4-", "prompt"),
        ("gpt-4$", "prompt"),
        ("gpt-4o-2024-05-13", "json_object"),
    ),
    # Every current Mistral model supports both modes except codestral-mamba.
    "mistral": (("codestral-mamba", "prompt"), ("open-codestral-mamba", "prompt")),
}


def structured_output_for(preset: ProviderPreset, model: str) -> StructuredOutput:
    for prefix, style in _MODEL_STRUCTURED_OUTPUT.get(preset.name, ()):
        exact = prefix.endswith("$")
        if (model == prefix[:-1]) if exact else model.startswith(prefix):
            return style
    return preset.structured_output


def _accepted_efforts(preset: ProviderPreset, model: str) -> tuple[Effort, ...]:
    for prefix, values in _MODEL_EFFORT_VALUES.items():
        if model.startswith(prefix):
            return tuple(v for v in values if v in preset.effort_values) or values
    return preset.effort_values


def _resolve_effort(settings: Settings, preset: ProviderPreset, model: str) -> Effort | None:
    configured = settings.generation_effort
    if configured == "off":
        return None
    if configured is not None:
        return configured
    if preset.default_effort is None:
        return None
    applies = (
        any(model.startswith(prefix) for prefix in preset.default_effort_models)
        if preset.default_effort_models
        else model == preset.default_model
    )
    return preset.default_effort if applies else None


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
    if (
        settings.generation_api_key is not None
        and preset.api_key_setting is not None
        and getattr(settings, preset.api_key_setting) is not None
        and getattr(settings, preset.api_key_setting).get_secret_value()
        != settings.generation_api_key.get_secret_value()
    ):
        # Fail before any key leaves the process: a stale GENERATION_API_KEY
        # from another provider would otherwise be sent to this endpoint.
        raise ProviderConfigError(
            f"Both GENERATION_API_KEY and {preset.api_key_setting.upper()} are set to "
            "different values; remove one"
        )

    effort = _resolve_effort(settings, preset, model)
    warnings: list[str] = []

    temperature = settings.generation_temperature
    if temperature is not None:
        if preset.kind is AdapterKind.ANTHROPIC and temperature > 1.0:
            raise ProviderConfigError(
                "GENERATION_TEMPERATURE must be between 0 and 1 for Anthropic"
            )
        if preset.name == "openai" and effort not in (None, "none"):
            raise ProviderConfigError(
                "OpenAI reasoning models reject GENERATION_TEMPERATURE unless "
                "GENERATION_EFFORT=none (or GENERATION_EFFORT=off for non-reasoning models)"
            )

    if effort is not None and preset.effort_param == "unsupported":
        warnings.append(f"GENERATION_EFFORT is not supported for provider {preset.name!r}; ignored")
        effort = None
    elif effort is not None and preset.effort_param == "reasoning_effort":
        clamped = clamp_effort(effort, _accepted_efforts(preset, model))
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
        structured_output=structured_output_for(preset, model),
    )


def resolve_auxiliary_llm(
    settings: Settings,
    *,
    provider: ProviderName,
    model: str,
    effort: Effort | Literal["off"] | None,
    max_output_tokens: int,
    setting_prefix: str,
    feature: str,
    users: str,
) -> ResolvedProvider:
    """A model used by a component other than the reply (decisions, the
    safety classifier), resolved like the reply model: same key, effort and
    structured-output rules.

    It uses the provider's own key setting (e.g. GROQ_API_KEY).
    GENERATION_API_KEY and GENERATION_BASE_URL apply only when the provider
    is also the generation provider, so the reply provider's key is never
    sent to another provider's endpoint. `setting_prefix` (e.g.
    "DECISION_LLM"), `feature` (e.g. "DECISION_PROVIDER=llm") and `users`
    (e.g. "decisions") only name things in error and warning messages.
    """
    preset = PRESETS[provider]
    shared = preset.name == settings.generation_provider
    if preset.name == "openai_compatible" and not shared:
        # Its endpoint is GENERATION_BASE_URL; without it the SDK would fall
        # back to environment defaults.
        raise ProviderConfigError(
            f"{setting_prefix}_PROVIDER=openai_compatible needs "
            "GENERATION_PROVIDER=openai_compatible (it uses GENERATION_BASE_URL)"
        )
    own_key = getattr(settings, preset.api_key_setting) if preset.api_key_setting else None
    if preset.requires_api_key and own_key is None and not (shared and settings.generation_api_key):
        key_var = (preset.api_key_setting or "generation_api_key").upper()
        raise ProviderConfigError(
            f"{feature} needs an API key for {setting_prefix}_PROVIDER={preset.name}: "
            f"set {key_var}"
        )
    resolved = resolve_provider(
        settings.model_copy(
            update={
                "generation_provider": preset.name,
                "generation_model": model,
                "generation_effort": effort,
                "generation_temperature": None,
                "generation_api_key": settings.generation_api_key if shared else None,
                "generation_base_url": settings.generation_base_url if shared else None,
                "max_output_tokens": max_output_tokens,
            }
        )
    )
    reply_model = settings.generation_model or PRESETS[settings.generation_provider].default_model
    if shared and resolved.model == reply_model:
        return replace(
            resolved,
            warnings=(
                *resolved.warnings,
                f"{setting_prefix}_MODEL is the reply model: {users} share its rate limits",
            ),
        )
    return resolved


def resolve_decision_llm(settings: Settings) -> ResolvedProvider:
    """The decision model for DECISION_PROVIDER=llm (see `resolve_auxiliary_llm`),
    from DECISION_LLM_PROVIDER, DECISION_LLM_MODEL, DECISION_LLM_EFFORT and
    DECISION_LLM_MAX_OUTPUT_TOKENS."""
    return resolve_auxiliary_llm(
        settings,
        provider=settings.decision_llm_provider,
        model=settings.decision_llm_model,
        effort=settings.decision_llm_effort,
        max_output_tokens=settings.decision_llm_max_output_tokens,
        setting_prefix="DECISION_LLM",
        feature="DECISION_PROVIDER=llm",
        users="decisions",
    )

from __future__ import annotations

import sys
from typing import Any, get_args

import pytest

from app.config import ProviderName
from app.llm.factory import build_llm_client
from app.llm.providers.groq_provider import GroqLLMClient
from app.llm.providers.openai_provider import OpenAICompatibleLLMClient
from app.llm.providers.registry import (
    PRESETS,
    AdapterKind,
    ProviderConfigError,
    clamp_effort,
    resolve_provider,
)
from tests.helpers import make_settings

NATIVE_KINDS = {AdapterKind.ANTHROPIC, AdapterKind.GROQ, AdapterKind.GEMINI, AdapterKind.FAKE}


def settings_for(provider: str, **overrides: Any) -> Any:
    keys = {
        "groq_api_key": "gsk-test",
        "openai_api_key": "sk-test",
        "gemini_api_key": "g-test",
        "xai_api_key": "xai-test",
        "dashscope_api_key": "ds-test",
        "deepseek_api_key": "ds-test",
        "openrouter_api_key": "or-test",
        "together_api_key": "tg-test",
        "mistral_api_key": "ms-test",
    }
    values: dict[str, Any] = {"generation_provider": provider, **keys}
    if provider == "openai_compatible":
        values |= {"generation_base_url": "http://localhost:8080/v1", "generation_model": "m"}
    values.update(overrides)
    return make_settings(**values)


def test_every_provider_name_has_a_preset() -> None:
    assert set(get_args(ProviderName)) == set(PRESETS)


def test_presets_are_internally_consistent() -> None:
    for preset in PRESETS.values():
        if preset.kind is AdapterKind.OPENAI_COMPATIBLE:
            assert preset.effort_param in (
                "reasoning_effort",
                "enable_thinking",
                "unsupported",
            ), preset.name
            assert preset.extra == "openai"
        else:
            assert preset.effort_param == "adapter", preset.name
        assert preset.send_openai_headers == (preset.name == "openai"), preset.name
        if preset.requires_api_key:
            assert preset.api_key_setting, preset.name
        assert preset.default_max_output_tokens >= 256


def test_groq_defaults_fit_the_free_tier() -> None:
    resolved = resolve_provider(settings_for("groq"))
    assert resolved.model == "openai/gpt-oss-120b"
    assert resolved.effort == "low"
    # A whole turn must fit inside the ~8K tokens/minute free-tier budget.
    assert resolved.max_output_tokens + resolved.max_context_tokens <= 8000


def test_overrides_win_over_preset() -> None:
    resolved = resolve_provider(
        settings_for(
            "groq",
            generation_model="qwen/qwen3.8-27b",
            generation_effort="none",
            max_output_tokens=1024,
            max_context_tokens=3000,
            generation_base_url="https://proxy.example/groq",
        )
    )
    assert resolved.model == "qwen/qwen3.8-27b"
    assert resolved.effort == "none"
    assert resolved.max_output_tokens == 1024
    assert resolved.max_context_tokens == 3000
    assert resolved.base_url == "https://proxy.example/groq"


def test_provider_key_and_generic_key_precedence() -> None:
    assert resolve_provider(settings_for("groq")).api_key == "gsk-test"
    generic = resolve_provider(
        make_settings(generation_provider="groq", generation_api_key="generic")
    )
    assert generic.api_key == "generic"


def test_missing_key_names_the_variable() -> None:
    with pytest.raises(ProviderConfigError, match="GROQ_API_KEY"):
        resolve_provider(make_settings(generation_provider="groq"))


def test_keyless_providers_resolve_without_key() -> None:
    ollama = resolve_provider(
        make_settings(generation_provider="ollama", generation_model="llama3")
    )
    assert ollama.api_key is None
    assert ollama.base_url == "http://localhost:11434/v1"


def test_model_required_when_preset_has_no_default() -> None:
    with pytest.raises(ProviderConfigError, match="GENERATION_MODEL"):
        resolve_provider(settings_for("openrouter"))


def test_openai_compatible_requires_base_url_and_model() -> None:
    with pytest.raises(ValueError, match="GENERATION_BASE_URL"):
        make_settings(generation_provider="openai_compatible")


def test_resolved_repr_never_shows_the_key() -> None:
    resolved = resolve_provider(settings_for("groq"))
    assert "gsk-test" not in repr(resolved)
    assert "api_key=set" in repr(resolved)


def test_unsupported_effort_is_dropped_with_warning() -> None:
    unsupported = [p for p in PRESETS.values() if p.effort_param == "unsupported"]
    if not unsupported:
        pytest.skip("no preset declares unsupported effort")
    preset = unsupported[0]
    resolved = resolve_provider(
        settings_for(preset.name, generation_effort="high", generation_model="m")
    )
    assert resolved.effort is None
    assert resolved.warnings


@pytest.mark.parametrize(
    "provider",
    [p.name for p in PRESETS.values() if p.kind is AdapterKind.OPENAI_COMPATIBLE],
)
def test_factory_builds_openai_compatible_clients(provider: str) -> None:
    settings = settings_for(provider, generation_model="some-model")
    client = build_llm_client(settings, resolve_provider(settings))
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.provider == provider


def test_factory_builds_groq_client() -> None:
    settings = settings_for("groq")
    client = build_llm_client(settings, resolve_provider(settings))
    assert isinstance(client, GroqLLMClient)
    assert client.provider == "groq"


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the "groq" extra not being installed.
    monkeypatch.setitem(sys.modules, "groq", None)
    monkeypatch.delitem(sys.modules, "app.llm.providers.groq_provider", raising=False)
    settings = settings_for("groq")
    with pytest.raises(ProviderConfigError, match="uv sync --extra groq"):
        build_llm_client(settings, resolve_provider(settings))


@pytest.mark.parametrize(
    ("effort", "allowed", "expected"),
    [
        ("max", ("none", "low", "medium", "high"), "high"),
        ("xhigh", ("none", "low", "medium", "high"), "high"),
        ("medium", ("none", "low", "high", "max"), "low"),
        ("low", ("none", "high"), "none"),
        ("medium", ("none", "high"), "high"),
        ("high", ("none", "low", "medium", "high"), "high"),
    ],
)
def test_clamp_effort(effort: str, allowed: tuple[str, ...], expected: str) -> None:
    assert clamp_effort(effort, allowed) == expected  # type: ignore[arg-type]


def test_xai_effort_is_clamped_with_warning() -> None:
    resolved = resolve_provider(settings_for("xai", generation_effort="max"))
    assert resolved.effort == "high"
    assert resolved.warnings


def test_presets_requiring_explicit_models() -> None:
    for name in ("deepseek", "openrouter", "together", "ollama"):
        with pytest.raises(ProviderConfigError, match="GENERATION_MODEL"):
            resolve_provider(settings_for(name))


def test_effort_off_omits_the_parameter() -> None:
    assert resolve_provider(settings_for("groq", generation_effort="off")).effort is None


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", None, "low"),
        ("anthropic", "claude-haiku-4-5", None),
        ("openai", None, "low"),
        ("openai", "gpt-4.1-mini", None),
        ("groq", None, "low"),
        ("groq", "openai/gpt-oss-20b", "low"),
        ("groq", "qwen/qwen3.8-27b", "low"),
        ("groq", "llama-3.1-8b-instant", None),
        ("gemini", "gemini-3.8-flash", "low"),
    ],
)
def test_default_effort_only_for_models_with_known_controls(
    provider: str, model: str | None, expected: str | None
) -> None:
    overrides = {"generation_model": model} if model else {}
    assert resolve_provider(settings_for(provider, **overrides)).effort == expected


def test_anthropic_rejects_temperature_above_one_at_startup() -> None:
    with pytest.raises(ProviderConfigError, match="between 0 and 1"):
        resolve_provider(settings_for("anthropic", generation_temperature=1.3))


def test_openai_temperature_requires_effort_none() -> None:
    with pytest.raises(ProviderConfigError, match="GENERATION_EFFORT=none"):
        resolve_provider(settings_for("openai", generation_temperature=0.7))
    ok = resolve_provider(
        settings_for("openai", generation_temperature=0.7, generation_effort="none")
    )
    assert ok.effort == "none"
    assert (
        resolve_provider(
            settings_for("openai", generation_model="gpt-4.1-mini", generation_temperature=0.7)
        ).effort
        is None
    )


def test_temperature_up_to_two_allowed_for_other_providers() -> None:
    assert resolve_provider(settings_for("groq", generation_temperature=1.6)).model


def test_conflicting_generic_and_provider_keys_fail_fast() -> None:
    with pytest.raises(ProviderConfigError, match="GROQ_API_KEY"):
        resolve_provider(settings_for("groq", generation_api_key="sk-ant-stale"))
    same = resolve_provider(settings_for("groq", generation_api_key="gsk-test"))
    assert same.api_key == "gsk-test"


def test_model_specific_effort_values() -> None:
    resolved = resolve_provider(
        settings_for("xai", generation_model="grok-4.7", generation_effort="none")
    )
    assert resolved.effort == "low"
    assert resolved.warnings
    assert (
        resolve_provider(
            settings_for("xai", generation_model="grok-4.3", generation_effort="none")
        ).effort
        == "none"
    )


def test_factory_passes_base_url_to_anthropic() -> None:
    settings = settings_for("anthropic", generation_base_url="https://gateway.example/anthropic")
    client = build_llm_client(settings, resolve_provider(settings))
    assert str(client._client.base_url).startswith("https://gateway.example/anthropic")  # type: ignore[attr-defined]

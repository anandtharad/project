"""Build the decision service from settings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import Settings
from app.decisions.llm_backend import DECISION_PROMPT_ID, LLMDecisionBackend
from app.decisions.questions import QUESTIONS, verify_locked
from app.decisions.service import DecisionBackend, DecisionService
from app.decisions.thresholds import calibration_key, thresholds_for
from app.llm.factory import build_llm_client
from app.llm.providers.registry import ProviderConfigError, resolve_decision_llm
from app.llm.schemas import GenerationParams
from app.observability import log_event
from app.prompting import PromptRegistry

if TYPE_CHECKING:
    import httpx2

logger = logging.getLogger("app.decisions")


def build_llm_decision_backend(settings: Settings) -> LLMDecisionBackend:
    """DECISION_PROVIDER=llm: a dedicated client for the decision model,
    resolved like the reply model (registry presets, keys, effort mapping)."""
    resolved = resolve_decision_llm(settings)
    for warning in resolved.warnings:
        log_event(logger, "config.warning", level=logging.WARNING, detail=f"decisions: {warning}")
    return LLMDecisionBackend(
        llm=build_llm_client(settings, resolved, max_retries=settings.decision_max_retries),
        llm_provider=resolved.preset.name,
        model=resolved.model,
        prompt=PromptRegistry().get(DECISION_PROMPT_ID, settings.decision_prompt_version),
        params=GenerationParams(effort=resolved.effort),
        max_output_tokens=resolved.max_output_tokens,
        timeout_seconds=settings.decision_timeout_seconds,
    )


def build_decision_backend(
    settings: Settings, *, http_client: httpx2.AsyncClient | None = None
) -> DecisionBackend | None:
    if settings.decision_provider == "off":
        return None
    if settings.decision_provider == "llm":
        return build_llm_decision_backend(settings)
    try:
        from app.decisions.backend import SystemOneBackend
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise ProviderConfigError(
            "The decision SDK is not installed. Run: uv sync --extra decisions"
        ) from exc
    if settings.decision_provider == "jev":
        key, base_url, model = settings.jev_api_key, settings.jev_base_url, settings.jev_model
    else:
        key, base_url, model = settings.laya_api_key, settings.laya_base_url, settings.laya_model
    assert key is not None  # enforced by Settings validation
    return SystemOneBackend(
        provider=settings.decision_provider,
        api_key=key.get_secret_value(),
        base_url=base_url,
        model=model,
        timeout_seconds=settings.decision_timeout_seconds,
        max_retries=settings.decision_max_retries,
        http_client=http_client,
    )


def build_decision_service(
    settings: Settings, *, backend: DecisionBackend | None = None
) -> DecisionService:
    verify_locked(QUESTIONS)
    if backend is None:
        backend = build_decision_backend(settings)
    key = calibration_key(settings)
    return DecisionService(
        provider=settings.decision_provider,
        backend=backend,
        questions=QUESTIONS,
        # Only thresholds measured for this exact provider, model and prompt.
        thresholds=thresholds_for(key),
        timeout_seconds=settings.decision_timeout_seconds,
        calibration_key=key,
    )

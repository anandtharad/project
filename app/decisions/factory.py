"""Build the decision service from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings
from app.decisions.questions import QUESTIONS, verify_locked
from app.decisions.service import DecisionBackend, DecisionService
from app.decisions.thresholds import THRESHOLDS
from app.llm.providers.registry import ProviderConfigError

if TYPE_CHECKING:
    import httpx2


def build_decision_backend(
    settings: Settings, *, http_client: httpx2.AsyncClient | None = None
) -> DecisionBackend | None:
    if settings.decision_provider == "off":
        return None
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
    return DecisionService(
        provider=settings.decision_provider,
        backend=backend,
        questions=QUESTIONS,
        thresholds=THRESHOLDS[settings.decision_provider],
        timeout_seconds=settings.decision_timeout_seconds,
    )

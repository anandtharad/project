"""Builds the safety layer from settings (SAFETY_* in app/config.py)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import ProviderName, Settings
from app.llm.base import LLMClient
from app.llm.providers.registry import ResolvedProvider, resolve_auxiliary_llm
from app.observability import log_event
from app.safety.analysis import SafetyAnalyzer
from app.safety.classifier import SafetyClassifier, build_safety_classifier
from app.safety.output import (
    SafetyOutputValidator,
    SemanticOutputChecker,
    build_semantic_output_checker,
)
from app.safety.policy import SafetyPolicy
from app.safety.resources import default_resources, normalize_region

logger = logging.getLogger("app.safety")


@dataclass
class SafetyLayer:
    analyzer: SafetyAnalyzer
    policy: SafetyPolicy
    validator: SafetyOutputValidator
    # Semantic check of replies on safety pathways (None: off or no classifier).
    output_checker: SemanticOutputChecker | None
    region: str

    async def aclose(self) -> None:
        if self.output_checker is not None:
            await self.output_checker.aclose()
        await self.analyzer.aclose()


def classifier_provider(settings: Settings) -> ProviderName | None:
    """The classifier's provider, or None when there is no classifier
    ("off", or "auto" without a Groq key)."""
    configured = settings.safety_classifier_provider
    if configured == "off":
        return None
    if configured == "auto":
        return "groq" if settings.groq_api_key is not None else None
    return configured


def resolve_safety_llm(settings: Settings, provider: ProviderName) -> ResolvedProvider:
    return resolve_auxiliary_llm(
        settings,
        provider=provider,
        model=settings.safety_classifier_model,
        effort=settings.safety_classifier_effort,
        max_output_tokens=settings.safety_classifier_max_output_tokens,
        setting_prefix="SAFETY_CLASSIFIER",
        feature="The safety classifier",
        users="safety checks",
    )


def build_classifier_from_settings(
    settings: Settings, *, llm: LLMClient | None = None
) -> SafetyClassifier | None:
    """The configured classifier, or None. `llm` replaces the client (tests)."""
    provider = classifier_provider(settings)
    if provider is None:
        return None
    resolved = resolve_safety_llm(settings, provider)
    for warning in resolved.warnings:
        log_event(logger, "config.warning", level=logging.WARNING, detail=f"safety: {warning}")
    classifier = build_safety_classifier(
        settings,
        resolved,
        policy_version=settings.safety_policy_version,
        timeout_seconds=settings.safety_classifier_timeout_seconds,
        llm=llm,
    )
    return classifier


def build_safety_layer(settings: Settings, *, llm: LLMClient | None = None) -> SafetyLayer | None:
    """None when SAFETY_ENABLED=false (refused in production)."""
    if not settings.safety_enabled:
        log_event(logger, "safety.disabled", level=logging.WARNING)
        return None
    classifier = build_classifier_from_settings(settings, llm=llm)
    if classifier is None:
        # Not an error: every turn runs in degraded mode and is recorded so.
        log_event(logger, "safety.classifier_off", level=logging.WARNING)
    checker: SemanticOutputChecker | None = None
    if classifier is not None and settings.safety_output_llm_check:
        provider = classifier_provider(settings)
        assert provider is not None
        checker = build_semantic_output_checker(
            settings,
            resolve_safety_llm(settings, provider),
            policy_version=settings.safety_output_policy_version,
            timeout_seconds=settings.safety_classifier_timeout_seconds,
            # Shares the classifier's client (one connection pool, closed once).
            llm=classifier.client,
        )
    directory = default_resources()
    return SafetyLayer(
        analyzer=SafetyAnalyzer(classifier, policy_version=settings.safety_policy_version),
        policy=SafetyPolicy(directory),
        validator=SafetyOutputValidator(directory),
        output_checker=checker,
        region=normalize_region(settings.safety_region),
    )

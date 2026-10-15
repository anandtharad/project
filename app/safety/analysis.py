"""Input analysis: patterns + classifier (+ decision signal) -> assessment.

Combination (design §2.4), in order of authority:

    category = max_severity(
        explicit pattern hits      (never downgraded by the classifier),
        classifier category        (only when its status is ok),
        self_harm                  (only when the decision signal is acted on),
    )

- Classifier ok: concern-only pattern hits do not override a classifier
  `none` or `distress` (fewer false positives) but are recorded.
- Classifier not ok (any failure, or disabled): degraded mode. The category
  is the most severe explicit hit, or `ambiguous_concern` if there is any
  concern hit, and `degraded` is set. With no hits at all the category is
  `none` but `assessed` is false: the turn proceeds with the ordinary policy
  (whose danger guidance is always present) and the record says
  "unassessed", never "safe".

The assessment holds labels, rule ids and statuses only, never text.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.llm.schemas import ChatMessage, ChatRole
from app.observability import log_event
from app.safety.classifier import (
    DEFAULT_POLICY_VERSION,
    DISABLED,
    Classifier,
    ClassifierResult,
    ClassifierStatus,
)
from app.safety.patterns import PATTERNS_VERSION, PatternResult, screen
from app.safety.taxonomy import Pathway, SafetyCategory, max_category, pathway_for

logger = logging.getLogger("app.safety")


@dataclass(frozen=True)
class SafetyAssessment:
    category: SafetyCategory
    pathway: Pathway
    # The classifier gave no label (failure or disabled): patterns only.
    degraded: bool
    # False when nothing could establish the category: degraded with no
    # pattern hit. Such a turn is "unassessed", never "safe".
    assessed: bool
    classifier_status: ClassifierStatus
    classifier_model: str | None
    # The classifier's own label (ok only), kept for evaluation.
    classifier_category: SafetyCategory | None
    pattern_rules: tuple[str, ...]
    policy_version: str
    patterns_version: str
    latency_ms: float
    decision_signal: bool = False

    def with_decision_signal(self) -> SafetyAssessment:
        """The assessment once a calibrated decision says the conversation mode
        is `safety`: at least self_harm (never lowers the category)."""
        category = max_category((self.category, SafetyCategory.SELF_HARM))
        return replace(
            self,
            category=category,
            pathway=pathway_for(category),
            assessed=True,
            decision_signal=True,
        )

    def summary(self) -> dict[str, object]:
        """Labels and statuses for provenance metadata and logs (no content)."""
        return {
            "category": self.category.value,
            "pathway": self.pathway.value,
            "degraded": self.degraded,
            "assessed": self.assessed,
            "classifier_status": self.classifier_status.value,
            "classifier_model": self.classifier_model,
            "pattern_rules": list(self.pattern_rules),
            "policy_version": self.policy_version,
            "patterns_version": self.patterns_version,
            "decision_signal": self.decision_signal,
        }


def combine(
    patterns: PatternResult,
    classifier: ClassifierResult,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
    decision_signal: bool = False,
    latency_ms: float = 0.0,
) -> SafetyAssessment:
    """The combination rule of the module docstring."""
    explicit = [hit.category for hit in patterns.explicit]
    if classifier.ok:
        category = max_category((*explicit, classifier.category))
        degraded = False
    else:
        concern = SafetyCategory.AMBIGUOUS_CONCERN if patterns.has_concern else None
        category = max_category((*explicit, concern))
        degraded = True
    assessment = SafetyAssessment(
        category=category,
        pathway=pathway_for(category),
        degraded=degraded,
        assessed=not degraded or category is not SafetyCategory.NONE,
        classifier_status=classifier.status,
        classifier_model=classifier.model_id,
        classifier_category=classifier.category if classifier.ok else None,
        pattern_rules=patterns.rule_ids,
        policy_version=policy_version,
        patterns_version=PATTERNS_VERSION,
        latency_ms=latency_ms,
    )
    return assessment.with_decision_signal() if decision_signal else assessment


def previous_user_message(history: Sequence[ChatMessage]) -> str | None:
    return next((m.content for m in reversed(history) if m.role is ChatRole.USER), None)


class SafetyAnalyzer:
    """Assesses every turn's latest message (patterns always, the classifier
    when configured). `assess` never raises: a classifier that fails, or is
    missing, gives degraded mode."""

    def __init__(
        self,
        classifier: Classifier | None,
        *,
        policy_version: str | None = None,
    ) -> None:
        self.classifier = classifier
        self.policy_version = policy_version or (
            classifier.policy_version if classifier is not None else DEFAULT_POLICY_VERSION
        )

    async def _classify(self, message: str, history: Sequence[ChatMessage]) -> ClassifierResult:
        if self.classifier is None:
            return DISABLED
        try:
            return await self.classifier.classify(message, history)
        except Exception as exc:
            # A bug in the classifier must not fail the turn or read as safe.
            log_event(
                logger,
                "safety.classifier_error",
                level=logging.ERROR,
                exc_type=type(exc).__name__,
            )
            return ClassifierResult(ClassifierStatus.UNAVAILABLE, model_id=self.classifier.model)

    async def assess(
        self,
        message: str,
        history: Sequence[ChatMessage] = (),
        *,
        decision_signal: bool = False,
    ) -> SafetyAssessment:
        """`history`: the earlier messages, oldest first, without `message`."""
        started = time.perf_counter()
        patterns = screen(message, previous_user_message(history))
        classified = await self._classify(message, history)
        assessment = combine(
            patterns,
            classified,
            policy_version=self.policy_version,
            decision_signal=decision_signal,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        log_event(logger, "safety.assessed", **assessment.summary())
        return assessment

    async def aclose(self) -> None:
        if self.classifier is not None:
            await self.classifier.aclose()

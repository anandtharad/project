"""Confidence thresholds per calibration identity (spec §7.7).

A threshold is the minimum certainty (choice/score: the returned confidence;
noul: the probability of the more likely outcome) at which the application
acts on a model's answer. Each value must come from an evaluation on labelled
development data and be confirmed on held-out data; docs/decisions.md records
the evidence.

A threshold belongs to the exact setup it was measured with, not just to a
provider: a Jev confidence, a Laya confidence and an LLM's normalised weights
are computed differently, and another model or decision prompt gives
differently distributed numbers. So thresholds are keyed by calibration
identity, and changing the model or the prompt version selects a different
(at first empty) entry instead of silently reusing another model's values:

- jev:  "jev:<JEV_MODEL>"
- laya: "laya:<LAYA_MODEL>"
- llm:  "llm:<DECISION_LLM_PROVIDER>:<DECISION_LLM_MODEL>:decision_llm@<DECISION_PROMPT_VERSION>"

A question missing for the active identity runs in shadow mode.
"""

from __future__ import annotations

from app.config import Settings
from app.decisions.llm_backend import DECISION_PROMPT_ID

# calibration key -> question key (id@version) -> threshold
THRESHOLDS: dict[str, dict[str, float]] = {}


def calibration_key(settings: Settings) -> str | None:
    """The identity thresholds are looked up by; None when decisions are off."""
    provider = settings.decision_provider
    if provider == "jev":
        return f"jev:{settings.jev_model}"
    if provider == "laya":
        return f"laya:{settings.laya_model}"
    if provider == "llm":
        return (
            f"llm:{settings.decision_llm_provider}:{settings.decision_llm_model}:"
            f"{DECISION_PROMPT_ID}@{settings.decision_prompt_version}"
        )
    return None


def thresholds_for(key: str | None) -> dict[str, float]:
    return dict(THRESHOLDS.get(key, {})) if key is not None else {}

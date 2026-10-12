"""Per-provider confidence thresholds (spec §7.7).

A threshold is the minimum certainty (choice/score: the returned confidence;
noul: the probability of the more likely outcome) at which the application
acts on a model's answer. Each value must come from an evaluation on labelled
development data and be confirmed on held-out data; docs/decisions.md records
the evidence. A question missing here runs in shadow mode for that provider.
"""

from __future__ import annotations

from app.decisions.schemas import DecisionProvider

THRESHOLDS: dict[DecisionProvider, dict[str, float]] = {
    "off": {},
    "jev": {},
    "laya": {},
}

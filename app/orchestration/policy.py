"""Response policy for a turn (spec §8.3, §9.2 step 9).

Deterministic code chooses how the next reply should approach the person's
message; the prompt then carries a fixed, versioned instruction for it (a
fragment of the companion prompt). Order of precedence:

1. safety_signal:    a calibrated decision says the conversation mode is
                     `safety`. The reply follows the prompt's danger guidance.
                     This is a signal, not the safety layer (Phase 5).
2. explicit_request: the message itself says what the person wants
                     ("I just need to vent", "what should I do?").
3. decision:         a calibrated response_strategy decision.
4. default:          no constraint; the model chooses the approach, as in
                     Phase 1. Uncalibrated or failed decisions end here, so
                     they can never make replies worse than the baseline.

The policy always allows a complete reply with neither advice nor a question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from app.decisions.questions import RESPONSE_STRATEGY
from app.decisions.schemas import DecisionSet


class Strategy(StrEnum):
    REFLECT = "reflect"
    VALIDATE = "validate"
    EXPLORE = "explore"
    CLARIFY = "clarify"
    SUMMARIZE = "summarize"
    PROVIDE_INFORMATION = "provide_information"
    PROBLEM_SOLVE = "problem_solve"
    PLAN = "plan"
    CHECK_IN = "check_in"


# Every decision option has a policy strategy, and vice versa.
assert set(RESPONSE_STRATEGY.options) == {s.value for s in Strategy}

PolicySource = Literal["safety_signal", "explicit_request", "decision", "default"]

_NO_ADVICE = re.compile(
    r"\b(?:don'?t|do not|no|not looking for|not asking for)\b[^.?!]{0,25}"
    r"\b(?:advice|solutions?|fix(?:ing)?|suggestions?|tips)\b"
    r"|\bjust (?:want|need) to (?:vent|rant|be heard|talk|get (?:it|this) off my chest)\b"
    r"|\bjust (?:listen|hear me out)\b",
    re.IGNORECASE,
)
_EXPLICIT: tuple[tuple[re.Pattern[str], Strategy], ...] = (
    (
        re.compile(r"\b(?:summari[sz]e|recap|sum (?:it|this|that|things) up)\b", re.I),
        Strategy.SUMMARIZE,
    ),
    (
        re.compile(
            r"\b(?:help me (?:make a |)plan|make (?:me )?a plan|plan (?:out )?my\b"
            r"|(?:a|my) (?:schedule|to-?do list) for)",
            re.I,
        ),
        Strategy.PLAN,
    ),
    (
        re.compile(
            r"\b(?:what should i do|what would you do|any (?:advice|suggestions|tips|ideas)"
            r"|(?:give|got) (?:me )?(?:some )?(?:advice|suggestions|tips)"
            r"|help me (?:decide|figure out|fix|solve|choose))\b",
            re.I,
        ),
        Strategy.PROBLEM_SOLVE,
    ),
)


@dataclass(frozen=True)
class ResponsePolicy:
    # None: no constraint, the model chooses the approach.
    strategy: Strategy | None
    advice_allowed: bool
    source: PolicySource
    safety_signal: bool = False

    @property
    def fragments(self) -> tuple[str, ...]:
        """Names of the prompt fragments that express this policy, in order."""
        if self.safety_signal:
            names = ["safety_signal"]
        else:
            names = [self.strategy.value if self.strategy else "open"]
        if not self.advice_allowed:
            names.append("no_advice")
        return tuple(names)

    def summary(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value if self.strategy else None,
            "advice_allowed": self.advice_allowed,
            "source": self.source,
            "safety_signal": self.safety_signal,
            "fragments": list(self.fragments),
        }


def explicit_request(message: str) -> tuple[Strategy | None, bool]:
    """(strategy the person asked for, whether advice is welcome)."""
    if _NO_ADVICE.search(message):
        return Strategy.VALIDATE, False
    for pattern, strategy in _EXPLICIT:
        if pattern.search(message):
            return strategy, True
    return None, True


def choose_policy(message: str, decisions: DecisionSet) -> ResponsePolicy:
    requested, advice_allowed = explicit_request(message)
    mode = decisions.acted_on("conversation_mode")
    if mode is not None and mode.value == "safety":
        return ResponsePolicy(
            strategy=None, advice_allowed=False, source="safety_signal", safety_signal=True
        )
    if requested is not None or not advice_allowed:
        return ResponsePolicy(
            strategy=requested, advice_allowed=advice_allowed, source="explicit_request"
        )
    decided = decisions.acted_on("response_strategy")
    if decided is not None:
        assert isinstance(decided.value, str)
        return ResponsePolicy(
            strategy=Strategy(decided.value), advice_allowed=True, source="decision"
        )
    return ResponsePolicy(strategy=None, advice_allowed=True, source="default")

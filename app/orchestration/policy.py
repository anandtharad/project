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

The default (open), reflect, validate and no-advice policies allow a complete
reply with neither advice nor a question; explore and clarify ask one
question, and problem_solve and plan offer options or steps. The prompt's
danger guidance always takes precedence over the chosen approach.

Explicit requests are matched conservatively, as first-person requests to
the companion: a mention of a word ("my boss asked me to summarize...") or a
statement of hopelessness ("there are no solutions left") is not a request.
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

_NEGATION = r"(?:don't|do not|didn't|did not|doesn't|does not|never|not|stop)"
_ADVICE_WORDS = r"(?:advice|solutions?|fix(?:es)?|suggestions?|tips)"
_NO_ADVICE = re.compile(
    rf"\b{_NEGATION}\s+(?:really\s+|even\s+)?"
    r"(?:want(?:ing)?|need(?:ing)?|looking for|ask(?:ed|ing)? for|giv(?:e|ing) me)\s+"
    rf"(?:any\s+|some\s+|your\s+)?{_ADVICE_WORDS}\b"
    r"|\bno (?:advice|suggestions|tips)(?: please| right now| for now)?\s*(?:[.!,]|$)"
    r"|\bjust (?:want|need) to (?:vent|rant|be heard|talk|get (?:it|this) off my chest)\b"
    r"(?!\s+(?:to|with)\b)"
    r"|\bjust (?:listen|hear me out)\b",
    re.IGNORECASE,
)
_EXPLICIT: tuple[tuple[re.Pattern[str], Strategy], ...] = (
    (
        re.compile(
            r"\b(?:(?:can|could|would|will) you (?:please )?"
            r"(?:summari[sz]e|recap|sum (?:it|this|that|things|everything) up)"
            r"|(?:please )?summari[sz]e (?:what (?:we|i)|this conversation|our|everything)"
            r"|give me a (?:quick |short )?(?:recap|summary))",
            re.I,
        ),
        Strategy.SUMMARIZE,
    ),
    (
        re.compile(
            r"\b(?:(?:can|could|would|will) you )?help me (?:make |write |put together )?"
            r"(?:a |my )?(?:plan|schedule|to-?do list)\b"
            r"|\b(?:can|could|would|will) you (?:make|write|draft) (?:me )?(?:a |my )?"
            r"(?:plan|schedule|to-?do list)\b"
            r"|\bhelp me plan\b",
            re.I,
        ),
        Strategy.PLAN,
    ),
    (
        re.compile(
            r"\bwhat (?:should|do you think) i (?:should )?do\b"
            r"|\bwhat would you do\b"
            # A question ("any tips for sleeping better?"), not a statement.
            r"|\b(?:any|got any|do you have any) (?:advice|suggestions|tips|ideas)\b[^.!?]*\?"
            r"|\b(?:can|could|would) you (?:please )?(?:give|offer) me (?:some )?"
            r"(?:advice|suggestions|tips)\b"
            r"|\bhelp me (?:decide|figure out|fix|solve|choose)\b",
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
            # The safety fragment carries its own limits on advice; adding
            # "no_advice" would forbid the crisis-line suggestion it requires.
            return ("safety_signal",)
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
    # Phones and word processors type curly apostrophes (U+2019 in "don\u2019t").
    message = message.replace("\u2019", "'").replace("\u2018", "'")
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
            strategy=None, advice_allowed=True, source="safety_signal", safety_signal=True
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

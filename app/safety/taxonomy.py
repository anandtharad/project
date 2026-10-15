"""Safety categories and response pathways (spec §12.3).

The categories are operational labels for routing a reply, not clinical
judgements. They are ordered by severity: when several apply, the most
severe wins (`max_category`). Idioms, fiction, news and questions about
other people's general facts are `none` ("this job is killing me lol",
"how do I kill a python process", "my character dies in chapter 3").
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class SafetyCategory(StrEnum):
    # Self-harm or harm to others that is planned, under way, or with
    # means, place or time stated.
    IMMINENT_DANGER = "imminent_danger"
    # Thoughts of suicide or self-harm, a wish to be dead, recent self-harm,
    # without imminence.
    SELF_HARM = "self_harm"
    # Intent or urge to seriously hurt another person.
    HARM_TO_OTHERS = "harm_to_others"
    # The person (or a child) is being abused, threatened or is unsafe at home.
    ABUSE = "abuse"
    # Asks for information that could seriously hurt self or others.
    DANGEROUS_REQUEST = "dangerous_request"
    # May signal risk but is unclear ("I can't do this anymore").
    AMBIGUOUS_CONCERN = "ambiguous_concern"
    # Signs of unhealthy reliance on the companion.
    DEPENDENCY = "dependency"
    # Emotional distress without risk indicators.
    DISTRESS = "distress"
    NONE = "none"


class Pathway(StrEnum):
    EMERGENCY = "emergency"
    CRISIS_SUPPORT = "crisis_support"
    HARM_TO_OTHERS = "harm_to_others"
    ABUSE_SUPPORT = "abuse_support"
    REFUSE_HARMFUL = "refuse_harmful"
    SAFETY_CHECK_IN = "safety_check_in"
    ENCOURAGE_CONNECTION = "encourage_connection"
    # The ordinary response policy (app/orchestration/policy.py).
    NORMAL = "normal"


# Higher is more severe. Declaration order of SafetyCategory is the same.
SEVERITY: dict[SafetyCategory, int] = {
    category: len(SafetyCategory) - 1 - index for index, category in enumerate(SafetyCategory)
}

PATHWAYS: dict[SafetyCategory, Pathway] = {
    SafetyCategory.IMMINENT_DANGER: Pathway.EMERGENCY,
    SafetyCategory.SELF_HARM: Pathway.CRISIS_SUPPORT,
    SafetyCategory.HARM_TO_OTHERS: Pathway.HARM_TO_OTHERS,
    SafetyCategory.ABUSE: Pathway.ABUSE_SUPPORT,
    # The refuse_harmful fragment follows crisis support when the request is
    # about the person themselves; a request that also states self-harm is
    # self_harm (more severe) and gets crisis_support directly.
    SafetyCategory.DANGEROUS_REQUEST: Pathway.REFUSE_HARMFUL,
    SafetyCategory.AMBIGUOUS_CONCERN: Pathway.SAFETY_CHECK_IN,
    SafetyCategory.DEPENDENCY: Pathway.ENCOURAGE_CONNECTION,
    SafetyCategory.DISTRESS: Pathway.NORMAL,
    SafetyCategory.NONE: Pathway.NORMAL,
}

# Every category has a pathway.
assert set(PATHWAYS) == set(SafetyCategory)


def pathway_for(category: SafetyCategory) -> Pathway:
    return PATHWAYS[category]


def max_category(categories: Iterable[SafetyCategory | None]) -> SafetyCategory:
    """The most severe category; `none` when there is none (None is skipped)."""
    present = [c for c in categories if c is not None]
    return max(present, key=SEVERITY.__getitem__, default=SafetyCategory.NONE)


def is_risk(category: SafetyCategory) -> bool:
    """Whether the category routes away from the ordinary policy."""
    return PATHWAYS[category] is not Pathway.NORMAL

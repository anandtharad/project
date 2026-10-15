"""Safety policy engine: assessment + region -> decision (spec §12.4).

One table, complete for every pathway, says how a reply on that pathway is
built: the companion prompt fragment that carries it, whether it overrides
the ordinary response strategy, what the reply must do (requirements, some
checked by the output validator), and which kinds of resources it shows.

For any pathway other than `normal` the ordinary strategy (`choose_policy`)
is bypassed and the reply uses the pathway fragment with the rendered
resources; an explicit "no advice" request never removes crisis resources.
`normal` leaves the turn to `choose_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.safety.analysis import SafetyAssessment
from app.safety.resources import (
    Resource,
    ResourceDirectory,
    ResourceKind,
    default_resources,
    normalize_region,
)
from app.safety.taxonomy import Pathway, SafetyCategory


class Requirement(StrEnum):
    # Checked by the output validator (missing_required):
    EMERGENCY_SERVICES = "emergency_services"  # contact emergency services now
    CRISIS_LINE = "crisis_line"  # a crisis line (from the resources given)
    TRUSTED_PERSON = "trusted_person"  # reach someone they trust
    # Checked by the output validator (dangerous_details):
    NO_METHOD_DETAILS = "no_method_details"  # no methods, doses or means
    # Carried by the pathway fragment (not checked deterministically):
    MOVE_AWAY_FROM_MEANS = "move_away_from_means"
    ASK_ABOUT_SAFETY = "ask_about_safety"  # ask directly whether they are safe
    STAY_IN_CONVERSATION = "stay_in_conversation"
    DECLINE_HARMFUL_DETAILS = "decline_harmful_details"
    BELIEVE_AND_PRIORITISE_SAFETY = "believe_and_prioritise_safety"
    ENCOURAGE_OTHER_SUPPORT = "encourage_other_support"


# Requirements the output validator checks as "the reply mentions ...".
HELP_REQUIREMENTS = frozenset({Requirement.EMERGENCY_SERVICES, Requirement.CRISIS_LINE})


@dataclass(frozen=True)
class PathwayRule:
    # The companion prompt fragment (None: the ordinary policy).
    fragment: str | None
    overrides_strategy: bool
    required: frozenset[Requirement]
    # Resource kinds shown, in order (a missing kind falls back, see
    # ResourceDirectory.select).
    resource_kinds: tuple[ResourceKind, ...]


_R = Requirement
_K = ResourceKind

PATHWAY_RULES: dict[Pathway, PathwayRule] = {
    Pathway.EMERGENCY: PathwayRule(
        fragment="safety_emergency",
        overrides_strategy=True,
        required=frozenset(
            {
                _R.EMERGENCY_SERVICES,
                _R.CRISIS_LINE,
                _R.TRUSTED_PERSON,
                _R.NO_METHOD_DETAILS,
                _R.MOVE_AWAY_FROM_MEANS,
                _R.STAY_IN_CONVERSATION,
            }
        ),
        resource_kinds=(_K.EMERGENCY, _K.CRISIS_LINE, _K.TEXT_LINE),
    ),
    Pathway.CRISIS_SUPPORT: PathwayRule(
        fragment="safety_crisis_support",
        overrides_strategy=True,
        required=frozenset(
            {
                _R.CRISIS_LINE,
                _R.TRUSTED_PERSON,
                _R.NO_METHOD_DETAILS,
                _R.ASK_ABOUT_SAFETY,
                _R.STAY_IN_CONVERSATION,
            }
        ),
        resource_kinds=(_K.CRISIS_LINE, _K.TEXT_LINE, _K.EMERGENCY),
    ),
    Pathway.HARM_TO_OTHERS: PathwayRule(
        fragment="safety_harm_to_others",
        overrides_strategy=True,
        required=frozenset(
            {_R.NO_METHOD_DETAILS, _R.DECLINE_HARMFUL_DETAILS, _R.MOVE_AWAY_FROM_MEANS}
        ),
        resource_kinds=(_K.CRISIS_LINE, _K.EMERGENCY),
    ),
    Pathway.ABUSE_SUPPORT: PathwayRule(
        fragment="safety_abuse_support",
        overrides_strategy=True,
        required=frozenset({_R.NO_METHOD_DETAILS, _R.BELIEVE_AND_PRIORITISE_SAFETY}),
        resource_kinds=(_K.ABUSE_SUPPORT, _K.EMERGENCY),
    ),
    Pathway.REFUSE_HARMFUL: PathwayRule(
        fragment="safety_refuse_harmful",
        overrides_strategy=True,
        required=frozenset({_R.NO_METHOD_DETAILS, _R.DECLINE_HARMFUL_DETAILS}),
        # Offered when the request is about the person themselves.
        resource_kinds=(_K.CRISIS_LINE,),
    ),
    Pathway.SAFETY_CHECK_IN: PathwayRule(
        fragment="safety_check_in",
        overrides_strategy=True,
        required=frozenset({_R.NO_METHOD_DETAILS, _R.ASK_ABOUT_SAFETY}),
        # No resources dump unless they say they are at risk.
        resource_kinds=(),
    ),
    Pathway.ENCOURAGE_CONNECTION: PathwayRule(
        fragment="safety_encourage_connection",
        overrides_strategy=True,
        required=frozenset({_R.NO_METHOD_DETAILS, _R.ENCOURAGE_OTHER_SUPPORT}),
        resource_kinds=(),
    ),
    Pathway.NORMAL: PathwayRule(
        fragment=None,
        overrides_strategy=False,
        required=frozenset(),
        resource_kinds=(),
    ),
}

assert set(PATHWAY_RULES) == set(Pathway)  # complete for every pathway


@dataclass(frozen=True)
class SafetyDecision:
    category: SafetyCategory
    pathway: Pathway
    # The companion prompt fragment for the pathway (None for normal).
    fragment: str | None
    overrides_strategy: bool
    required: frozenset[Requirement]
    resources: tuple[Resource, ...]
    region: str  # an ISO code, or "" for the generic resources

    @property
    def resource_lines(self) -> tuple[str, ...]:
        """The resources as text lines for the reply guidance."""
        return tuple(r.line for r in self.resources)

    def summary(self) -> dict[str, object]:
        return {
            "pathway": self.pathway.value,
            "fragment": self.fragment,
            "overrides_strategy": self.overrides_strategy,
            "required": sorted(r.value for r in self.required),
            "resources": [r.id for r in self.resources],
            "region": self.region,
        }


class SafetyPolicy:
    def __init__(self, directory: ResourceDirectory | None = None) -> None:
        self.directory = directory or default_resources()

    def decide(self, assessment: SafetyAssessment, region: str) -> SafetyDecision:
        """An unknown region raises `UnknownRegionError` (checked at startup)."""
        code = normalize_region(region)
        rule = PATHWAY_RULES[assessment.pathway]
        return SafetyDecision(
            category=assessment.category,
            pathway=assessment.pathway,
            fragment=rule.fragment,
            overrides_strategy=rule.overrides_strategy,
            required=rule.required,
            resources=self.directory.select(code, rule.resource_kinds),
            region=code,
        )

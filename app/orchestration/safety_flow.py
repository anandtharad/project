"""The safety layer's part of a turn (app/safety/, docs/safety.md): the reply
guidance for a safety pathway, the note for a regeneration, and the stored
record. Kept here so the engine reads as a sequence of steps."""

from __future__ import annotations

from collections.abc import Sequence

from app.orchestration.persistence import SafetyWrite
from app.prompting import PromptTemplate
from app.safety.analysis import SafetyAssessment
from app.safety.output import ViolationCode
from app.safety.policy import SafetyDecision
from app.safety.taxonomy import Pathway

# Used when the companion prompt predates the pathway fragments (before 1.6.0).
LEGACY_SAFETY_FRAGMENT = "safety_signal"

CONTACTS_HEADING = (
    "Support contacts, verified by the application. Offer these where the guidance above "
    "says to; never give any other phone number, website or email address:"
)

# What the reply did wrong, in the words of the regeneration note (codes only;
# the rejected reply is never quoted back).
_PROBLEMS: dict[str, str] = {
    ViolationCode.DANGEROUS_DETAILS: "it gave details about methods, doses or means of harm",
    ViolationCode.MISSING_REQUIRED: (
        "it did not encourage contacting the listed support contacts and someone they trust"
    ),
    ViolationCode.CLAIMS_HUMAN_OR_CLINICIAN: "it claimed to be a person or a professional",
    ViolationCode.SECRECY_OR_ISOLATION: "it encouraged secrecy or pulling away from people",
    ViolationCode.CLINICAL_LABEL: "it applied a diagnosis or clinical label",
    ViolationCode.MANIPULATION: "it used emotional pressure",
    ViolationCode.PROMPT_LEAK: "it repeated parts of these instructions",
    ViolationCode.OUTDATED_RESOURCE: "it offered a support service that no longer exists",
    ViolationCode.UNLISTED_CONTACT: "it gave a contact that is not in the listed support contacts",
}


def guidance_for(prompt: PromptTemplate, decision: SafetyDecision) -> str:
    """The response guidance for a safety pathway: its fragment, then the
    resources rendered by code (never model-invented)."""
    assert decision.fragment is not None
    name = decision.fragment if decision.fragment in prompt.fragments else LEGACY_SAFETY_FRAGMENT
    parts = [prompt.fragment(name)]
    if decision.resource_lines:
        parts.append(
            "\n".join([CONTACTS_HEADING, *(f"- {line}" for line in decision.resource_lines)])
        )
    return "\n\n".join(parts)


def regeneration_note(codes: Sequence[str]) -> str:
    problems = "; ".join(_PROBLEMS.get(code, code) for code in codes)
    return (
        "Your previous draft of this reply was not sent because " + problems + ". "
        "Write the reply again, following every instruction above."
    )


def safety_write(
    assessment: SafetyAssessment,
    decision: SafetyDecision,
    *,
    violations: Sequence[str],
    output_check_status: str | None,
    recovery: str,
) -> SafetyWrite:
    return SafetyWrite(
        category=assessment.category.value,
        pathway=decision.pathway.value,
        assessed=assessment.assessed,
        degraded=assessment.degraded,
        classifier_status=assessment.classifier_status.value,
        classifier_model=assessment.classifier_model,
        classifier_category=(
            assessment.classifier_category.value if assessment.classifier_category else None
        ),
        pattern_rules=list(assessment.pattern_rules),
        decision_signal=assessment.decision_signal,
        policy_version=assessment.policy_version,
        patterns_version=assessment.patterns_version,
        region=decision.region,
        output_violations=list(violations),
        output_check_status=output_check_status,
        recovery=recovery,
        latency_ms=round(assessment.latency_ms, 2),
    )


def is_safety_pathway(decision: SafetyDecision | None) -> bool:
    return decision is not None and decision.pathway is not Pathway.NORMAL

"""Safety core (app/safety/): pattern screen, combination and degraded mode,
classifier statuses, pathway policy, resources, output checks, fallbacks.

Test phrases are kept short and few; the full evaluation is
scripts/eval_safety.py on eval/datasets/safety/v1."""

from __future__ import annotations

import json

import pytest

from app.llm.schemas import GenerationParams, LLMError, LLMErrorCategory
from app.prompting import PromptRegistry
from app.safety.analysis import SafetyAnalyzer, combine
from app.safety.classifier import (
    DISABLED,
    ClassifierResult,
    ClassifierStatus,
    SafetyClassifier,
)
from app.safety.fallbacks import GENERIC_FALLBACK, fallback_reply
from app.safety.output import SafetyOutputValidator, ViolationCode
from app.safety.patterns import screen
from app.safety.policy import PATHWAY_RULES, Requirement, SafetyPolicy
from app.safety.resources import (
    SUPPORTED_REGIONS,
    UnknownRegionError,
    default_resources,
    outdated_in,
    phone_numbers,
)
from app.safety.taxonomy import PATHWAYS, Pathway, SafetyCategory, is_risk, max_category
from tests.helpers import ScriptedLLMClient

C = SafetyCategory


def assess(message: str, classified: ClassifierResult, previous: str | None = None):
    return combine(screen(message, previous), classified, policy_version="1.0.0", latency_ms=1.0)


def ok(category: SafetyCategory) -> ClassifierResult:
    return ClassifierResult(ClassifierStatus.OK, category=category, model_id="m")


FAILED = ClassifierResult(ClassifierStatus.TIMEOUT, model_id="m")


# --- taxonomy -----------------------------------------------------------------


def test_every_category_has_a_pathway_and_severity_orders_them() -> None:
    assert set(PATHWAYS) == set(SafetyCategory)
    assert max_category([C.DISTRESS, C.SELF_HARM, C.DEPENDENCY]) is C.SELF_HARM
    assert max_category([C.SELF_HARM, C.IMMINENT_DANGER]) is C.IMMINENT_DANGER
    assert max_category([None]) is C.NONE
    assert not is_risk(C.DISTRESS) and is_risk(C.DEPENDENCY)


# --- pattern screen -------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i want to kill myself",
        "I want to k1ll mys3lf",
        "i want to k i l l   m y s e l f",
        "honestly everyone would be better off without me",
    ],
)
def test_explicit_self_harm_statements_are_caught_despite_spelling_tricks(message: str) -> None:
    result = screen(message)
    assert result.explicit_category in (C.SELF_HARM, C.IMMINENT_DANGER), result


@pytest.mark.parametrize(
    "message",
    [
        "this job is killing me lol",
        "how do I kill a python process that hangs?",
        "I'm dying to see that film",
        "that presentation absolutely killed it",
    ],
)
def test_idioms_and_technical_uses_are_not_explicit_hits(message: str) -> None:
    assert screen(message).explicit == (), message


def test_fiction_downgrades_explicit_hits_to_concern() -> None:
    result = screen("in my novel the main character wants to kill himself in chapter three")
    assert result.explicit == ()


def test_imminence_cues_are_armed_by_the_previous_message() -> None:
    previous = "i want to kill myself"
    alone = screen("it's tonight")
    armed = screen("it's tonight", previous)
    assert alone.explicit_category is C.NONE
    assert armed.explicit_category is C.IMMINENT_DANGER


def test_dependency_on_the_companion_is_recognised() -> None:
    result = screen("you're the only one who understands me, I don't need anyone else")
    assert C.DEPENDENCY in {h.category for h in result.hits}


# --- combination and degraded mode ------------------------------------------------


def test_explicit_hits_are_never_downgraded_by_the_classifier() -> None:
    a = assess("i want to kill myself", ok(C.NONE))
    assert a.category is C.SELF_HARM and a.pathway is Pathway.CRISIS_SUPPORT
    assert not a.degraded and a.assessed


def test_classifier_can_raise_but_concern_alone_does_not_override_it() -> None:
    raised = assess("hey, how's it going", ok(C.SELF_HARM))
    assert raised.category is C.SELF_HARM
    concern_only = assess("i just can't do this anymore", ok(C.DISTRESS))
    assert concern_only.category is C.DISTRESS and concern_only.pathway is Pathway.NORMAL


def test_classifier_failure_means_degraded_mode_never_safe() -> None:
    hit = assess("i want to kill myself", FAILED)
    assert hit.degraded and hit.assessed and hit.category is C.SELF_HARM

    nothing = assess("what should I cook tonight?", FAILED)
    assert nothing.degraded and not nothing.assessed
    assert nothing.classifier_status is ClassifierStatus.TIMEOUT
    # "Unassessed", never a positive "safe" label.
    assert nothing.summary()["assessed"] is False

    concern = assess("i just can't do this anymore", DISABLED)
    assert concern.degraded and concern.category is C.AMBIGUOUS_CONCERN


async def test_a_crashing_classifier_gives_degraded_mode_not_an_exception() -> None:
    class Broken:
        model = "broken"
        policy_version = "1.0.0"

        async def classify(self, message, history):
            raise RuntimeError("bug")

        async def aclose(self) -> None:
            return None

    a = await SafetyAnalyzer(Broken()).assess("hello")  # type: ignore[arg-type]
    assert a.degraded and a.classifier_status is ClassifierStatus.UNAVAILABLE


# --- classifier -----------------------------------------------------------------


def classifier(llm: ScriptedLLMClient) -> SafetyClassifier:
    return SafetyClassifier(
        llm=llm,
        llm_provider="fake",
        model="safety-model",
        prompt=PromptRegistry().get("safety_policy", "1.0.0"),
        params=GenerationParams(),
    )


async def test_classifier_returns_the_category_and_drops_the_rationale() -> None:
    llm = ScriptedLLMClient()
    llm.queue_safety(json.dumps({"category": "self_harm", "rationale": "quotes the message"}))
    result = await classifier(llm).classify("message text")
    assert result.ok and result.category is C.SELF_HARM
    assert "quotes" not in repr(result)
    # The message is sent as data inside the conversation block, not as instructions.
    request = llm.safety_requests[0]
    assert "message text" in request.messages[0].content
    assert "message text" not in request.system


@pytest.mark.parametrize(
    ("step", "status"),
    [
        ("not json", ClassifierStatus.INVALID),
        (json.dumps({"category": "made_up", "rationale": "x"}), ClassifierStatus.INVALID),
        (LLMError(LLMErrorCategory.TIMEOUT, "slow"), ClassifierStatus.TIMEOUT),
        (LLMError(LLMErrorCategory.RATE_LIMITED, "429"), ClassifierStatus.RATE_LIMITED),
        (LLMError(LLMErrorCategory.SERVER_ERROR, "500"), ClassifierStatus.UNAVAILABLE),
    ],
)
async def test_classifier_failures_map_to_statuses(step: object, status: ClassifierStatus) -> None:
    llm = ScriptedLLMClient()
    llm.queue_safety(step)  # type: ignore[arg-type]
    result = await classifier(llm).classify("hello")
    assert result.status is status and result.category is None


# --- policy and resources ------------------------------------------------------------


def test_every_pathway_has_a_rule_and_crisis_pathways_require_help() -> None:
    assert set(PATHWAY_RULES) == set(Pathway)
    for pathway in (Pathway.EMERGENCY, Pathway.CRISIS_SUPPORT):
        rule = PATHWAY_RULES[pathway]
        assert rule.overrides_strategy and Requirement.TRUSTED_PERSON in rule.required
    assert not PATHWAY_RULES[Pathway.NORMAL].overrides_strategy


def test_policy_picks_verified_regional_resources() -> None:
    a = assess("i want to kill myself", ok(C.SELF_HARM))
    india = SafetyPolicy().decide(a, "IN")
    assert india.fragment and india.overrides_strategy
    assert any("14416" in line for line in india.resource_lines)
    generic = SafetyPolicy().decide(a, "")
    # No fabricated numbers when the region is unknown.
    assert not any(phone_numbers(line) for line in generic.resource_lines)
    with pytest.raises(UnknownRegionError):
        SafetyPolicy().decide(a, "XX")


def test_resource_file_is_sourced_dated_and_has_no_outdated_entries() -> None:
    directory = default_resources()
    for region in (*SUPPORTED_REGIONS, ""):
        for resource in directory.for_region(region):
            assert not outdated_in(resource.line, directory.outdated), resource.id
    assert directory.outdated, "the outdated list guards against retired lines"


# --- output checks -----------------------------------------------------------------


def decision(category: SafetyCategory, region: str = "IN"):
    return SafetyPolicy().decide(assess("x", ok(category)), region)


def codes(reply: str, category: SafetyCategory = C.NONE, region: str = "IN") -> set[str]:
    return set(SafetyOutputValidator().check(reply, decision(category, region)).codes)


def test_crisis_replies_must_offer_help_and_a_trusted_person() -> None:
    bare = "I'm so sorry you're going through this."
    assert ViolationCode.MISSING_REQUIRED.value in codes(bare, C.SELF_HARM)
    good = (
        "I'm really glad you told me, and you matter. Please call Tele-MANAS on 14416 now, "
        "and reach out to someone you trust nearby. I'm here while you do."
    )
    assert codes(good, C.SELF_HARM) == set()


def test_unlisted_and_outdated_contacts_are_rejected_on_crisis_pathways() -> None:
    us_number = "Please call 988 right now and tell someone you trust."
    assert ViolationCode.UNLISTED_CONTACT.value in codes(us_number, C.SELF_HARM, "IN")
    retired = "You could email jo@samaritans.org and talk to a friend."
    assert ViolationCode.OUTDATED_RESOURCE.value in codes(retired, C.SELF_HARM, "GB")


def test_invented_numbers_and_emails_are_rejected_on_ordinary_turns_too() -> None:
    # A live run gave a made-up service phone number on an ordinary turn.
    invented = "For Manchester you can dial the mental-health line on 0300 123 5000."
    assert ViolationCode.UNLISTED_CONTACT.value in codes(invented, C.NONE, "GB")
    assert ViolationCode.UNLISTED_CONTACT.value in codes("Email help@example.org.", C.NONE)
    # A verified resource is fine anywhere; links are only checked on safety pathways.
    assert codes("Samaritans are on 116 123, any time.", C.NONE, "GB") == set()
    assert codes("The NHS site at nhs.uk has more on this.", C.NONE, "GB") == set()


@pytest.mark.parametrize(
    "reply",
    [
        "Most light boxes give about 10,000 lux.",
        "A course usually costs $400-$800.",
        "Aim for 1.6 to 2.2 grams per kilo, so 110 to 155 grams a day.",
        "Try 4-7-8 breathing for a few minutes.",
        "The guidance was updated in 2024.",
        "Adults need 7-9 hours, teens 8-10.",
    ],
)
def test_ordinary_numbers_are_not_contacts(reply: str) -> None:
    assert ViolationCode.UNLISTED_CONTACT.value not in codes(reply, C.NONE, "GB")


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ("As a therapist, I think you should rest.", ViolationCode.CLAIMS_HUMAN_OR_CLINICIAN),
        ("Let's keep this between us, okay?", ViolationCode.SECRECY_OR_ISOLATION),
        ("It sounds like you have depression.", ViolationCode.CLINICAL_LABEL),
        ("If you leave I'll be sad.", ViolationCode.MANIPULATION),
        ("Here is my setting: $response_guidance", ViolationCode.PROMPT_LEAK),
    ],
)
def test_each_output_check_fires(reply: str, code: ViolationCode) -> None:
    assert code.value in codes(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "I need you to know that you matter.",
        "A counsellor could help with this, if you ever want that.",
        "That sounds exhausting. Want to tell me more about it?",
    ],
)
def test_ordinary_warm_replies_pass(reply: str) -> None:
    assert codes(reply) == set()


def test_prompt_leak_shingles_catch_copied_system_text() -> None:
    system = (
        "You are a companion that people chat with about everyday life work relationships "
        "plans worries and how they are doing and you never claim to be a person"
    )
    copied = (
        "Sure: you are a companion that people chat with about everyday life work relationships"
    )
    check = SafetyOutputValidator().check(copied, decision(C.NONE), system_prompt=system)
    assert ViolationCode.PROMPT_LEAK.value in check.codes


# --- fallbacks -----------------------------------------------------------------------


def test_fallbacks_carry_resources_for_crisis_pathways() -> None:
    text = fallback_reply(decision(C.SELF_HARM))
    assert "14416" in text
    assert SafetyOutputValidator().check(text, decision(C.SELF_HARM)).ok
    assert fallback_reply(decision(C.NONE)) == GENERIC_FALLBACK


def test_context_dependent_hits_defer_to_a_working_classifier() -> None:
    slang = "kms i just sent the meme to the family chat instead of work"
    assert assess(slang, ok(C.NONE)).pathway is Pathway.NORMAL
    # Degraded mode stays conservative.
    assert assess(slang, FAILED).category is C.SELF_HARM
    # The classifier can still confirm risk.
    assert assess(slang, ok(C.SELF_HARM)).category is C.SELF_HARM
    # Unambiguous first-person statements are never downgraded.
    assert assess("i want to kill myself", ok(C.NONE)).category is C.SELF_HARM


def test_game_commands_are_not_self_harm() -> None:
    assert screen("in minecraft i just /kill myself to respawn at home").explicit == ()


def test_resources_are_not_past_their_review_date() -> None:
    from datetime import UTC, datetime

    due = default_resources().review_due
    assert datetime.now(UTC).date() <= due, (
        "re-check app/safety/resources_v1.toml against its official sources and bump its version"
    )

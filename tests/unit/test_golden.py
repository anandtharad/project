"""Golden-scenario checks (app/evaluation/golden.py): each check can fail,
skips are never counted as passes, and the scenario file is valid."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.golden import (
    Expect,
    Observed,
    Scenario,
    ScenarioOutcome,
    Step,
    check_step,
    load_scenarios,
    summarize,
)

DATASET = (
    Path(__file__).resolve().parents[2] / "eval" / "datasets" / "golden" / "v1" / "scenarios.jsonl"
)


def outcomes(expect: Expect, seen: Observed, mode: str = "live") -> dict[str, str]:
    return {c.name: c.outcome for c in check_step(expect, seen, mode)}  # type: ignore[arg-type]


def test_pipeline_checks_pass_and_fail() -> None:
    expect = Expect(safety_pathway="crisis_support", explicit=True, knowledge="no_lookup")
    good = Observed(201, safety_pathway="crisis_support", knowledge_retrieve=False)
    bad = Observed(201, safety_pathway=None, knowledge_retrieve=True)
    assert outcomes(expect, good, "deterministic") == {
        "status": "pass",
        "citations_resolve": "pass",
        "safety_pathway": "pass",
        "knowledge": "pass",
    }
    result = outcomes(expect, bad, "deterministic")
    assert result["safety_pathway"] == "fail" and result["knowledge"] == "fail"


def test_classifier_dependent_checks_are_skipped_without_a_model() -> None:
    expect = Expect(safety_pathway="any_safety", max_questions=1)
    result = outcomes(expect, Observed(201), "deterministic")
    assert result["safety_pathway"] == "skip" and result["reply"] == "skip"
    live = outcomes(expect, Observed(201, reply="Are you ok? Really? Truly?"), "live")
    assert live["safety_pathway"] == "fail" and live["max_questions"] == "fail"


def test_reply_checks() -> None:
    reply = "It sounds like a lot.\n- try this\n- or this\nAccording to the CDC [S1]."
    expect = Expect(
        no_lists=True,
        no_stock_phrases=True,
        mentions_none=["according to"],
        mentions_any=["priya"],
        cites=True,
        max_words=5,
    )
    result = outcomes(expect, Observed(201, reply=reply, message="hi"))
    assert {k: v for k, v in result.items() if k != "status"} == {
        "citations_resolve": "fail",  # a marker with no source returned
        "no_lists": "fail",
        "no_stock_phrases": "fail",
        "max_words": "fail",
        "mentions_any": "fail",
        "mentions_none": "fail",
        "cites": "fail",
    }


def test_status_probe_and_scoring() -> None:
    assert outcomes(Expect(status=503), Observed(503)) == {"status": "pass"}
    assert outcomes(Expect(), Observed(200, probe_ok=False)) == {"probe": "fail"}
    scenario = Scenario(
        id="s-1", category="casual", split="dev", description="x", steps=[Step(user="hi")]
    )
    skipped = ScenarioOutcome(
        scenario,
        [
            (0, c)
            for c in check_step(Expect(max_questions=1), Observed(201), "deterministic")
            if c.name == "reply"
        ],
    )
    assert skipped.passed is None  # all skipped: unchecked, never passed
    assert summarize([skipped])["checked"] == 0


def test_steps_and_categories_are_validated() -> None:
    with pytest.raises(ValidationError):
        Step(user="hi", action="new_conversation")
    with pytest.raises(ValidationError):
        Step(action="fail_next")
    with pytest.raises(ValidationError):
        Scenario(id="x", category="nonsense", split="dev", description="", steps=[Step(user="a")])


def test_the_committed_scenarios_load() -> None:
    scenarios = load_scenarios(DATASET.read_text(encoding="utf-8").splitlines())
    assert scenarios and len({s.id for s in scenarios}) == len(scenarios)

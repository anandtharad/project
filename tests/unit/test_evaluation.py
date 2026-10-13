from __future__ import annotations

from pathlib import Path

import pytest

from app.decisions.questions import CONVERSATION_MODE, KNOWLEDGE_RETRIEVAL_USEFUL, QUESTIONS
from app.evaluation.decisions import (
    Example,
    Prediction,
    _auroc,
    _ece,
    choice_metrics,
    load_examples,
    noul_metrics,
    predict,
    select_threshold,
    summarize,
    threshold_sweep,
)
from app.llm.schemas import ChatMessage, ChatRole
from tests.helpers import ScriptedDecisionBackend, choice

DATASET = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "decisions" / "v1"


def example(i: int, mode: str, retrieval: bool = False) -> Example:
    return Example(
        id=f"x-{i}",
        conversation=[ChatMessage(role=ChatRole.USER, content=f"message {i}")],
        labels={
            "conversation_mode": mode,
            "response_strategy": "reflect",
            "knowledge_retrieval_useful": retrieval,
        },
        difficulty="clear",
        family="x",
    )


def predicted(i: int, mode: str | None, confidence: float = 0.9, noul: float = 0.2) -> Prediction:
    answers = {
        "conversation_mode": choice(CONVERSATION_MODE, mode, confidence) if mode else None,
        "knowledge_retrieval_useful": {"type": "noul", "noul": noul},
    }
    return Prediction(example_id=f"x-{i}", answers=answers, latency_ms=10.0, model_id="m")


def test_ece_and_auroc() -> None:
    assert _ece([0.9, 0.9], [True, True]) == pytest.approx(0.1)
    assert _ece([], []) is None
    assert _auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert _auroc([0.5, 0.5], [True, False]) == 0.5
    assert _auroc([0.5], [True]) is None


def test_threshold_is_the_lowest_meeting_accuracy_and_coverage() -> None:
    confidences = [0.95] * 12 + [0.5] * 20
    correct = [True] * 11 + [False] + [False] * 20
    sweep = threshold_sweep(confidences, correct)
    assert select_threshold(sweep) == 0.55  # first threshold above the 0.5 bucket
    assert select_threshold(threshold_sweep([0.9] * 5, [True] * 5)) is None  # too few acted on


def test_choice_metrics_against_gold_and_baseline() -> None:
    examples = [example(0, "listening"), example(1, "listening"), example(2, "casual")]
    predictions = [predicted(0, "listening"), predicted(1, "casual"), predicted(2, None)]
    m = choice_metrics(CONVERSATION_MODE, examples, predictions, threshold=0.5)
    assert m["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["valid_answers"] == 2
    assert m["majority_baseline"] == {"label": "listening", "accuracy": pytest.approx(0.6667)}
    assert m["confusion"] == {"listening": {"listening": 1, "casual": 1}, "casual": {"None": 1}}
    assert m["at_given_threshold"]["acted"] == 2


def test_noul_metrics() -> None:
    examples = [example(0, "casual", True), example(1, "casual", False)]
    predictions = [predicted(0, "casual", noul=0.8), predicted(1, "casual", noul=0.3)]
    m = noul_metrics(KNOWLEDGE_RETRIEVAL_USEFUL, examples, predictions)
    assert m["accuracy_at_0_5"] == 1.0 and m["auroc"] == 1.0
    assert m["brier"] == pytest.approx((0.2**2 + 0.3**2) / 2)


async def test_predict_validates_answers_and_records_failures() -> None:
    answers = {
        "conversation_mode": choice(CONVERSATION_MODE, "casual"),
        "response_strategy": {"type": "choice", "choice": "not-an-option"},
        "knowledge_retrieval_useful": {"type": "noul", "noul": 0.1},
    }
    backend = ScriptedDecisionBackend(answers)
    (prediction,) = await predict(backend, QUESTIONS, [example(0, "casual")])
    assert prediction.answers["conversation_mode"] is not None
    assert prediction.answers["response_strategy"] is None  # invalid answers are not scored
    assert backend.calls[0]["latest_message_from_person"] == "message 0"
    summary = summarize(QUESTIONS, [example(0, "casual")], [prediction])
    assert summary["questions"][CONVERSATION_MODE.key]["accuracy"] == 1.0


def test_dataset_files_parse_and_cover_every_label() -> None:
    dev, test = load_examples(DATASET / "dev.jsonl"), load_examples(DATASET / "test.jsonl")
    assert len(dev) == len(test) == 125
    assert not {e.id for e in dev} & {e.id for e in test}
    for e in dev + test:
        assert e.conversation[-1].role is ChatRole.USER
        assert e.labels["conversation_mode"] in CONVERSATION_MODE.options
    assert {e.labels["conversation_mode"] for e in dev + test} >= set(CONVERSATION_MODE.options)


def test_policy_effect_helpers() -> None:
    from app.evaluation.policy_effect import Sample, parse_label, summarize_effect, wilson_interval

    options = ("validate", "problem_solve", "explore")
    assert parse_label(" Problem_Solve.\n", options) == "problem_solve"
    assert parse_label("problem solve", options) == "problem_solve"
    assert parse_label("I think validate", options) is None
    low, high = wilson_interval(9, 10)
    assert 0.55 < low < 0.6 and 0.98 < high <= 1.0
    samples = [
        Sample("a", "validate", "That makes sense.", "validate"),
        Sample("a", "problem_solve", "You could try a list?", "problem_solve"),
        Sample("a", "explore", "What feels hardest?", "validate"),
        Sample("a", "open", "Hm.", "explore"),
    ]
    summary = summarize_effect(samples, options)
    assert summary["judge_recovery_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["per_strategy"]["problem_solve"]["advice_wording_rate"] == 1.0
    assert summary["per_strategy"]["explore"]["asks_question_rate"] == 1.0
    assert summary["chance"] == pytest.approx(1 / 3, abs=1e-3)


def test_effect_examples_are_open_ended_without_explicit_requests() -> None:
    from app.evaluation.policy_effect import OPEN_ENDED_MODES, select_examples
    from app.orchestration.policy import explicit_request

    chosen = select_examples(load_examples(DATASET / "test.jsonl"), 10)
    assert len(chosen) == 10
    for e in chosen:
        assert e.labels["conversation_mode"] in OPEN_ENDED_MODES
        assert explicit_request(e.conversation[-1].content) == (None, True)

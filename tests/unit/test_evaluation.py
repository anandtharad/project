from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from app.decisions.backend import BackendAnswers, DecisionBackendError
from app.decisions.llm_backend import DECISION_PROMPT_ID, LLMDecisionBackend
from app.decisions.questions import (
    CONVERSATION_MODE,
    KNOWLEDGE_RETRIEVAL_USEFUL,
    QUESTIONS,
    RESPONSE_STRATEGY,
)
from app.decisions.schemas import FallbackReason, QuestionSpec
from app.evaluation.decisions import (
    Example,
    Pacer,
    Prediction,
    RunStopped,
    _auroc,
    _ece,
    cache_file_name,
    cache_identity,
    choice_metrics,
    load_cache,
    load_examples,
    merge_predictions,
    noul_metrics,
    predict,
    result_file_name,
    run_identity,
    safe_name,
    save_cache,
    select_threshold,
    selected_thresholds,
    summarize,
    threshold_sweep,
    thresholds_source_problems,
)
from app.llm.schemas import ChatMessage, ChatRole, GenerationParams, LLMError, LLMErrorCategory
from app.prompting import PromptRegistry
from tests.helpers import ScriptedDecisionBackend, ScriptedLLMClient, choice, make_settings

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval" / "datasets" / "decisions" / "v1"


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


def test_failed_requests_count_against_threshold_coverage() -> None:
    examples = [example(i, "listening") for i in range(4)]
    predictions = [predicted(0, "listening", confidence=0.95)] + [
        Prediction(example_id=f"x-{i}", answers={"conversation_mode": None}) for i in (1, 2, 3)
    ]
    m = choice_metrics(CONVERSATION_MODE, examples, predictions, threshold=0.9)
    assert m["at_given_threshold"] == {
        "threshold": 0.9,
        "acted": 1,
        "coverage": 0.25,  # 1 of 4 turns, not 1 of 1 valid answer
        "accuracy": 1.0,
    }


# --- pacing, rate limits, run identity, caching (scripts/eval_decisions.py) ----


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


async def test_pacer_keeps_a_pause_and_a_token_budget() -> None:
    clock = FakeClock()
    pacer = Pacer(pause_seconds=2.0, tokens_per_minute=4000, clock=clock, sleep=clock.sleep)
    await pacer.wait()  # nothing sent yet
    assert clock.slept == []
    pacer.record(1500)
    await pacer.wait()  # 1500 + 1500 expected fits; only the pause
    assert clock.slept == [2.0]
    pacer.record(1500)
    clock.now += 2.0
    await pacer.wait()  # 3000 used + 1500 expected > 4000: wait for the first to expire
    assert clock.slept[-1] == pytest.approx(60.0 - 4.0)


class RateLimitedBackend(ScriptedDecisionBackend):
    """Raises the queued errors first, then answers."""

    def __init__(self, answers: dict[str, Any], errors: list[DecisionBackendError]) -> None:
        super().__init__(answers)
        self.errors = errors

    async def ask(self, state: Any, questions: list[QuestionSpec]) -> BackendAnswers:
        if self.errors:
            self.calls.append(dict(state))
            raise self.errors.pop(0)
        return await super().ask(state, questions)


def rate_limited(retry_after: float | None = None, quota: bool = False) -> DecisionBackendError:
    return DecisionBackendError(
        FallbackReason.RATE_LIMITED,
        "limit",
        status_code=429,
        retry_after_seconds=retry_after,
        quota_exhausted=quota,
    )


ANSWERS = {
    "conversation_mode": choice(CONVERSATION_MODE, "casual"),
    "knowledge_retrieval_useful": {"type": "noul", "noul": 0.1},
}


async def test_a_rate_limit_is_waited_out_not_recorded_as_a_failure() -> None:
    clock = FakeClock()
    backend = RateLimitedBackend(ANSWERS, [rate_limited(7.0), rate_limited(None)])
    (prediction,) = await predict(backend, QUESTIONS, [example(0, "casual")], sleep=clock.sleep)
    assert clock.slept == [7.0, 20.0]  # retry-after, then the second backoff step
    assert prediction.error is None and prediction.answers["conversation_mode"] is not None


@pytest.mark.parametrize(
    "error", [rate_limited(quota=True), rate_limited(retry_after=600.0)], ids=["quota", "long"]
)
async def test_a_quota_or_long_wait_stops_the_run(error: DecisionBackendError) -> None:
    kept: list[Prediction] = []
    backend = RateLimitedBackend(ANSWERS, [])
    examples = [example(0, "casual"), example(1, "casual")]

    async def second_is_limited(state: Any, questions: list[QuestionSpec]) -> BackendAnswers:
        if state["latest_message_from_person"] == "message 1":
            raise error
        return await ScriptedDecisionBackend.ask(backend, state, questions)

    backend.ask = second_is_limited  # type: ignore[method-assign]
    with pytest.raises(RunStopped) as info:
        await predict(backend, QUESTIONS, examples, on_prediction=kept.append)
    assert info.value.answered == 1
    assert [p.example_id for p in kept] == ["x-0"]  # nothing recorded for x-1


async def test_other_failures_are_still_recorded() -> None:
    error = DecisionBackendError(FallbackReason.TIMEOUT, "slow")
    (prediction,) = await predict(
        ScriptedDecisionBackend(error=error), QUESTIONS, [example(0, "casual")]
    )
    assert prediction.error == "timeout"


def llm_identity(model: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    backend = LLMDecisionBackend(
        llm=ScriptedLLMClient(),
        llm_provider="groq",
        model=model,
        prompt=PromptRegistry().get(DECISION_PROMPT_ID, "1.0.0"),
        params=GenerationParams(effort="low"),
        max_output_tokens=1024,
        timeout_seconds=5,
    )
    return run_identity(backend, QUESTIONS)


def test_run_identity_records_prompt_effort_and_output_limit() -> None:
    identity = llm_identity()
    assert identity["provider"] == "llm" and identity["requested_model"] == "openai/gpt-oss-20b"
    assert identity["decision_llm"] == {
        "provider": "groq",
        "effort": "low",
        "max_output_tokens": 1024,
    }
    prompt = identity["decision_prompt"]
    assert prompt["key"] == "decision_llm@1.0.0"
    assert prompt["checksum"] == PromptRegistry().get(DECISION_PROMPT_ID, "1.0.0").checksum
    assert len(prompt["request_sha256"]) == 64
    assert identity["questions"] == {q.key: q.checksum for q in QUESTIONS}
    laya = run_identity(ScriptedDecisionBackend(), QUESTIONS)
    assert laya["decision_llm"] is None and laya["decision_prompt"] is None


def test_result_file_names_are_safe_on_every_os() -> None:
    assert safe_name("openai/gpt-oss-20b") == "openai_gpt-oss-20b"
    assert safe_name("a:b\\c d") == "a_b_c_d"
    assert safe_name("../..") == "_"
    name = result_file_name(llm_identity(), "dev", "20261014T000000")
    assert name == (
        "decisions-llm-groq-openai_gpt-oss-20b-decision_llm_1.0.0-dev-20261014T000000.json"
    )
    assert "/" not in name and ":" not in name
    laya = run_identity(ScriptedDecisionBackend(), QUESTIONS)
    # Unchanged for laya/jev (eval/results/decisions-laya-english-*.json).
    assert result_file_name(laya, "test", "T") == "decisions-laya-english-test-T.json"


def dev_result(identity: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {"split": "dev", **identity, **changes}


def test_thresholds_from_refuses_a_different_identity() -> None:
    identity = llm_identity()
    assert thresholds_source_problems(dev_result(identity), identity) == []
    other_model = llm_identity("openai/gpt-oss-120b")
    assert any("requested_model" in p for p in thresholds_source_problems(other_model, identity))
    edited_prompt = dev_result(
        identity, decision_prompt={**identity["decision_prompt"], "checksum": "0" * 64}
    )
    assert any("decision_prompt" in p for p in thresholds_source_problems(edited_prompt, identity))
    edited_question = dev_result(identity, questions={CONVERSATION_MODE.key: "0" * 64})
    assert any("questions" in p for p in thresholds_source_problems(edited_question, identity))
    other_effort = dev_result(identity, decision_llm={**identity["decision_llm"], "effort": "high"})
    assert any("decision_llm" in p for p in thresholds_source_problems(other_effort, identity))
    laya = run_identity(ScriptedDecisionBackend(), QUESTIONS)
    assert any("provider" in p for p in thresholds_source_problems(dev_result(laya), identity))
    test_file = {**dev_result(identity), "split": "test"}
    assert any("dev only" in p for p in thresholds_source_problems(test_file, identity))


def test_a_laya_dev_file_from_before_these_fields_still_applies() -> None:
    old = json.loads(
        next((ROOT / "eval" / "results").glob("decisions-laya-english-dev-*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert "decision_prompt" not in old
    laya = run_identity(ScriptedDecisionBackend(), QUESTIONS)
    assert thresholds_source_problems(old, laya) == []
    assert selected_thresholds(old) == {}  # no Laya question reached the target


def test_cache_round_trip_and_identity_check(tmp_path: Path) -> None:
    cache_id = cache_identity(llm_identity(), dataset_sha256="abc", split="dev")
    path = tmp_path / cache_file_name(cache_id)
    assert path.name.startswith("decisions-llm-groq-openai_gpt-oss-20b-")
    assert load_cache(path, cache_id) == {}
    saved = {"x-0": predicted(0, "casual"), "x-1": Prediction(example_id="x-1", error="timeout")}
    save_cache(path, cache_id, saved)
    assert load_cache(path, cache_id) == saved
    other = cache_identity(llm_identity("openai/gpt-oss-120b"), dataset_sha256="abc", split="dev")
    assert cache_file_name(other) != path.name
    with pytest.raises(ValueError, match="different run"):
        load_cache(path, other)


def test_merge_keeps_example_order_prefers_new_answers_and_lists_missing() -> None:
    examples = [example(i, "casual") for i in range(4)]
    cached = {"x-0": predicted(0, "casual"), "x-2": Prediction(example_id="x-2", error="timeout")}
    new = [predicted(2, "listening"), predicted(1, "casual")]
    merged, missing = merge_predictions(examples, cached, new)
    assert [p.example_id for p in merged] == ["x-0", "x-1", "x-2"]
    assert merged[2].error is None  # the new answer replaced the cached failure
    assert missing == ["x-3"]


# --- the script, end to end with a scripted model ------------------------------


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "eval_decisions", ROOT / "scripts" / "eval_decisions.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision_reply() -> str:
    return json.dumps(
        {
            "conversation_mode": {
                o: (70 if o == "listening" else 5) for o in CONVERSATION_MODE.options
            },
            "response_strategy": {
                o: (60 if o == "validate" else 10) for o in RESPONSE_STRATEGY.options
            },
            "knowledge_retrieval_useful": {"probability_yes": 20},
        }
    )


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module = load_script()
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "results" / ".cache")
    settings = make_settings(decision_provider="llm", groq_api_key="gsk-test-not-real")
    monkeypatch.setattr(module, "Settings", lambda: settings)
    module.llm = ScriptedLLMClient()

    def backend(_: Any) -> LLMDecisionBackend:
        return LLMDecisionBackend(
            llm=module.llm,
            llm_provider="groq",
            model="openai/gpt-oss-20b",
            prompt=PromptRegistry().get(DECISION_PROMPT_ID, "1.0.0"),
            params=GenerationParams(effort="low"),
            max_output_tokens=1024,
            timeout_seconds=5,
        )

    monkeypatch.setattr(module, "build_decision_backend", backend)
    return module


def test_a_stopped_run_is_cached_and_resumed_into_one_result(script: Any, tmp_path: Path) -> None:
    daily_limit = LLMError(LLMErrorCategory.RATE_LIMITED, "limit", retry_after_seconds=3600)
    script.llm.queue_decision(decision_reply(), decision_reply(), daily_limit)
    assert script.main(["--split", "dev", "--limit", "4"]) == script.STOPPED
    results = tmp_path / "results"
    assert not list(results.glob("decisions-*.json"))  # no partial result file
    (cache_file,) = (results / ".cache").glob("decisions-llm-groq-openai_gpt-oss-20b-*.json")
    assert len(json.loads(cache_file.read_text(encoding="utf-8"))["predictions"]) == 2

    script.llm = ScriptedLLMClient()
    script.llm.queue_decision(decision_reply(), decision_reply())
    assert script.main(["--split", "dev", "--limit", "4", "--resume"]) == 0
    assert len(script.llm.decision_requests) == 2  # only the two unanswered examples
    (result_file,) = results.glob("decisions-llm-groq-openai_gpt-oss-20b-decision_llm_1.0.0-dev-*")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert len(result["predictions"]) == 4 and result["metrics"]["failed_requests"] == {}
    assert result["calibration_key"] == "llm:groq:openai/gpt-oss-20b:decision_llm@1.0.0"
    assert result["decision_prompt"]["key"] == "decision_llm@1.0.0"
    assert result["decision_llm"] == {
        "provider": "groq",
        "effort": "low",
        "max_output_tokens": 1024,
    }
    assert result["run"]["from_cache"] == 2 and result["run"]["resumed"] is True
    assert result["served_models"] == ["scripted-model"]


def test_start_continues_a_split_from_the_cache(script: Any, tmp_path: Path) -> None:
    script.llm.queue_decision(decision_reply(), decision_reply())
    assert script.main(["--split", "dev", "--limit", "2"]) == 0
    script.llm = ScriptedLLMClient()
    script.llm.queue_decision(decision_reply())
    assert script.main(["--split", "dev", "--start", "2", "--limit", "1"]) == 0
    assert len(script.llm.decision_requests) == 1
    # The newest result covers examples 0-2: two from the cache, one new.
    newest = max((tmp_path / "results").glob("decisions-*.json"), key=lambda p: p.name)
    result = json.loads(newest.read_text(encoding="utf-8"))
    assert len(result["predictions"]) == 3 and result["run"]["from_cache"] == 2
    # --start without the earlier examples in the cache cannot produce a result.
    script.llm = ScriptedLLMClient()
    script.llm.queue_decision(decision_reply())
    assert script.main(["--split", "test", "--start", "5", "--limit", "1"]) == script.STOPPED


def test_thresholds_from_a_mismatched_dev_file_is_refused_before_any_request(
    script: Any, tmp_path: Path
) -> None:
    laya_dev = next((ROOT / "eval" / "results").glob("decisions-laya-english-dev-*.json"))
    assert script.main(["--split", "test", "--thresholds-from", str(laya_dev)]) == 1
    assert script.llm.decision_requests == []

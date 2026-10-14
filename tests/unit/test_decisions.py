"""Decision layer through the real typesafe-sdk, with no network access.

httpx2.MockTransport answers every request, so these tests pin the exact
wire request, answer validation, thresholds, and every fallback path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from app.decisions.backend import DecisionBackendError, SystemOneBackend
from app.decisions.factory import build_decision_service
from app.decisions.questions import (
    CONVERSATION_MODE,
    KNOWLEDGE_RETRIEVAL_USEFUL,
    QUESTIONS,
    RESPONSE_STRATEGY,
    QuestionLockError,
    verify_locked,
)
from app.decisions.schemas import FallbackReason, QuestionSpec, ValidationStatus
from app.decisions.service import DecisionService, build_decision_state, validate_answer
from app.llm.schemas import ChatMessage, ChatRole
from tests.helpers import make_settings

Handler = Callable[[httpx2.Request], httpx2.Response]
STATE = {"latest_message_from_person": "I just need to vent about work."}


def choice_answer(question: QuestionSpec, choice: str, confidence: float = 0.9) -> dict[str, Any]:
    rest = (1.0 - 0.9) / (len(question.options) - 1)
    probabilities = {label: (0.9 if label == choice else rest) for label in question.options}
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def valid_answers() -> dict[str, Any]:
    return {
        "conversation_mode": choice_answer(CONVERSATION_MODE, "listening"),
        "response_strategy": choice_answer(RESPONSE_STRATEGY, "validate", confidence=0.8),
        "knowledge_retrieval_useful": {"type": "noul", "noul": 0.1},
    }


def ok(answers: dict[str, Any], *, model: str = "jev-1.13.0") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": 300, "output_tokens": 9},
        },
        headers={"x-typesafe-request-id": "req-123"},
    )


class Recorder:
    def __init__(self, response: httpx2.Response | Handler) -> None:
        self.requests: list[httpx2.Request] = []
        self._response = response

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._response(request) if callable(self._response) else self._response


def make_backend(handler: Handler, *, provider: Any = "jev", retries: int = 0) -> SystemOneBackend:
    return SystemOneBackend(
        provider=provider,
        api_key="ts-test-key",
        base_url="https://api.typesafe.test",
        model="jev-1.13.0",
        timeout_seconds=5,
        max_retries=retries,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def make_service(
    handler: Handler, *, thresholds: dict[str, float] | None = None, timeout: float = 5
) -> DecisionService:
    return DecisionService(
        provider="jev",
        backend=make_backend(handler),
        questions=QUESTIONS,
        thresholds=thresholds or {},
        timeout_seconds=timeout,
    )


async def test_request_is_the_documented_system_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ambient SDK variables must never redirect the call or swap the key.
    monkeypatch.setenv("TYPESAFE_API_KEY", "ambient-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://attacker.test")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
    recorder = Recorder(ok(valid_answers()))
    service = make_service(recorder)
    await service.decide(STATE)
    await service.aclose()

    (sent,) = recorder.requests
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.typesafe.test/v1/systemone"
    assert sent.headers["authorization"] == "Bearer ts-test-key"
    body = json.loads(sent.content)
    assert body["state"] == STATE
    assert body["model"] == "jev-1.13.0"
    assert body["questions"]["conversation_mode"] == {
        "type": "choice",
        "instructions": CONVERSATION_MODE.instructions,
        "criteria": CONVERSATION_MODE.criteria,
    }
    assert body["questions"]["knowledge_retrieval_useful"]["type"] == "noul"
    assert set(body) == {"state", "model", "questions"}


async def test_valid_answers_run_in_shadow_mode_without_thresholds() -> None:
    service = make_service(Recorder(ok(valid_answers())))
    decisions = await service.decide(STATE)
    strategy = decisions.results["response_strategy"]
    assert strategy.validation_status is ValidationStatus.VALID
    assert strategy.returned_value == "validate"
    assert strategy.value == RESPONSE_STRATEGY.fallback
    assert strategy.fallback and strategy.fallback_reason is FallbackReason.NOT_CALIBRATED
    assert strategy.confidence == 0.8
    assert strategy.model_id == "jev-1.13.0"
    assert decisions.request_id == "req-123"
    assert decisions.acted_on("response_strategy") is None


async def test_thresholds_decide_what_the_application_acts_on() -> None:
    thresholds = {
        RESPONSE_STRATEGY.key: 0.7,  # confidence 0.8 clears it
        CONVERSATION_MODE.key: 0.95,  # confidence 0.9 does not
        KNOWLEDGE_RETRIEVAL_USEFUL.key: 0.85,  # noul 0.1 -> certainty 0.9
    }
    decisions = await make_service(Recorder(ok(valid_answers())), thresholds=thresholds).decide(
        STATE
    )
    acted = decisions.acted_on("response_strategy")
    assert acted is not None and acted.value == "validate" and not acted.fallback
    mode = decisions.results["conversation_mode"]
    assert mode.fallback_reason is FallbackReason.LOW_CONFIDENCE
    assert mode.value == "other" and mode.returned_value == "listening"
    retrieval = decisions.results["knowledge_retrieval_useful"]
    assert not retrieval.fallback and retrieval.value == pytest.approx(0.1)
    assert retrieval.confidence is None  # Jev returns none for a noul; never fabricated


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (CONVERSATION_MODE, {**choice_answer(CONVERSATION_MODE, "listening"), "choice": "angry"}),
        (CONVERSATION_MODE, {**choice_answer(CONVERSATION_MODE, "listening"), "confidence": 1.5}),
        (
            CONVERSATION_MODE,
            {**choice_answer(CONVERSATION_MODE, "listening"), "probabilities": {"listening": 1.0}},
        ),
        (
            CONVERSATION_MODE,
            {
                **choice_answer(CONVERSATION_MODE, "listening"),
                "probabilities": dict.fromkeys(CONVERSATION_MODE.options, 0.5),
            },
        ),
        # The chosen label must be the most probable one.
        (
            RESPONSE_STRATEGY,
            {**choice_answer(RESPONSE_STRATEGY, "validate"), "choice": "reflect"},
        ),
        (CONVERSATION_MODE, {"type": "noul", "noul": 0.5}),
        (KNOWLEDGE_RETRIEVAL_USEFUL, {"type": "noul", "noul": 1.2}),
        (KNOWLEDGE_RETRIEVAL_USEFUL, {"type": "noul", "noul": "yes"}),
    ],
)
def test_answers_that_do_not_fit_the_question_are_rejected(
    question: QuestionSpec, answer: dict[str, Any]
) -> None:
    assert validate_answer(question, answer) is None


def test_score_answers_are_checked_against_the_rubric() -> None:
    question = QuestionSpec(
        id="urgency",
        version="1.0.0",
        kind="score",
        objective="test",
        instructions="How urgent?",
        criteria=["can wait", "soon", "now"],
        fallback=0,
        downstream="test",
    )
    good = {
        "type": "score",
        "score": 1.2,
        "confidence": 0.7,
        "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
    }
    assert validate_answer(question, good) == (1.2, {"0": 0.1, "1": 0.6, "2": 0.3}, 0.7)
    assert validate_answer(question, {**good, "score": 2.5}) is None
    assert validate_answer(question, {**good, "probabilities": {"0": 0.5, "1": 0.5}}) is None


async def test_invalid_and_missing_answers_fall_back_individually() -> None:
    answers = valid_answers()
    answers["conversation_mode"]["choice"] = "not-an-option"
    del answers["knowledge_retrieval_useful"]
    decisions = await make_service(
        Recorder(ok(answers)), thresholds={RESPONSE_STRATEGY.key: 0.5}
    ).decide(STATE)
    mode = decisions.results["conversation_mode"]
    assert mode.validation_status is ValidationStatus.INVALID
    assert mode.fallback_reason is FallbackReason.INVALID_ANSWER and mode.value == "other"
    retrieval = decisions.results["knowledge_retrieval_useful"]
    assert retrieval.validation_status is ValidationStatus.MISSING
    assert retrieval.fallback_reason is FallbackReason.MISSING_ANSWER
    # The other question is unaffected.
    assert decisions.acted_on("response_strategy") is not None


async def test_extra_fields_from_a_non_strict_laya_server_are_ignored() -> None:
    answers = valid_answers()
    answers["response_strategy"] |= {"x_jev_confidence": 0.7, "action": {"act_probability": 1.0}}
    body = {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": 1, "output_tokens": 0, "truncated": False},
        "routing": {"model": "english"},
    }
    decisions = await make_service(Recorder(httpx2.Response(200, json=body))).decide(STATE)
    strategy = decisions.results["response_strategy"]
    assert strategy.validation_status is ValidationStatus.VALID
    assert strategy.model_id == "laya-rl-agent"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx2.Response(401, json={"detail": "bad key"}), FallbackReason.REJECTED),
        (
            httpx2.Response(422, json={"detail": [{"loc": ["body"], "msg": "x"}]}),
            FallbackReason.REJECTED,
        ),
        (httpx2.Response(429, json={"detail": "slow down"}), FallbackReason.RATE_LIMITED),
        (httpx2.Response(529, json={"detail": "overloaded"}), FallbackReason.UNAVAILABLE),
        (httpx2.Response(500, text="boom"), FallbackReason.UNAVAILABLE),
        (httpx2.Response(200, json={"answers": {}}), FallbackReason.INVALID_RESPONSE),
        (httpx2.Response(200, text="not json"), FallbackReason.INVALID_RESPONSE),
    ],
)
async def test_service_failures_fall_back_for_every_question(
    response: httpx2.Response, reason: FallbackReason
) -> None:
    decisions = await make_service(Recorder(response)).decide(STATE)
    for question in QUESTIONS:
        result = decisions.results[question.id]
        assert result.fallback and result.fallback_reason is reason
        assert result.value == question.fallback
        assert result.validation_status is ValidationStatus.MISSING


async def test_rate_limit_wait_and_token_counts_are_passed_on() -> None:
    limited = httpx2.Response(429, json={"detail": "slow down"}, headers={"retry-after": "3"})
    with pytest.raises(DecisionBackendError) as info:
        await make_backend(Recorder(limited)).ask(STATE, list(QUESTIONS))
    assert info.value.reason is FallbackReason.RATE_LIMITED
    assert info.value.retry_after_seconds == pytest.approx(3.0)
    raw = await make_backend(Recorder(ok(valid_answers()))).ask(STATE, list(QUESTIONS))
    assert (raw.input_tokens, raw.output_tokens) == (300, 9)


async def test_connection_failure_falls_back() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    decisions = await make_service(refuse).decide(STATE)
    assert decisions.results["conversation_mode"].fallback_reason is FallbackReason.UNAVAILABLE


async def test_slow_service_is_cut_off_at_the_decision_timeout() -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(5)
        return ok(valid_answers())

    service = DecisionService(
        provider="laya",
        backend=SystemOneBackend(
            provider="laya",
            api_key="k",
            base_url="http://laya.test",
            model="english",
            timeout_seconds=30,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(slow)),
        ),
        questions=QUESTIONS,
        thresholds={},
        timeout_seconds=0.05,
    )
    decisions = await service.decide(STATE)
    assert decisions.results["response_strategy"].fallback_reason is FallbackReason.TIMEOUT


async def test_provider_off_uses_fallbacks_without_a_backend() -> None:
    service = build_decision_service(make_settings(decision_provider="off"))
    decisions = await service.decide(STATE)
    assert all(r.fallback_reason is FallbackReason.DISABLED for r in decisions.results.values())
    assert decisions.provider == "off"


def test_decision_providers_require_keys() -> None:
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        make_settings(decision_provider="jev")
    with pytest.raises(ValueError, match="LAYA_API_KEY"):
        make_settings(decision_provider="laya")
    settings = make_settings(decision_provider="jev", jev_api_key="ts-x")
    assert settings.jev_model == "jev-1.13.0"


def test_released_questions_are_locked() -> None:
    verify_locked(QUESTIONS)
    edited = RESPONSE_STRATEGY.model_copy(update={"instructions": "Something else?"})
    with pytest.raises(QuestionLockError, match="edited after release"):
        verify_locked((edited,))


def test_question_answer_spaces_are_validated() -> None:
    with pytest.raises(ValueError, match="fallback must be one of the options"):
        QuestionSpec.model_validate({**CONVERSATION_MODE.model_dump(), "fallback": "nope"})
    with pytest.raises(ValueError, match="exactly 'true' and 'false'"):
        QuestionSpec.model_validate(
            {**KNOWLEDGE_RETRIEVAL_USEFUL.model_dump(), "criteria": {"yes": "a", "no": "b"}}
        )
    assert "safety_response" not in RESPONSE_STRATEGY.options


def test_decision_state_puts_the_latest_message_first_and_bounds_history() -> None:
    history = [
        ChatMessage(role=ChatRole.USER if i % 2 == 0 else ChatRole.ASSISTANT, content=f"m{i}" * 400)
        for i in range(9)
    ]
    state = build_decision_state(history, {"topic": "work"})
    assert next(iter(state)) == "latest_message_from_person"
    assert state["latest_message_from_person"] == history[-1].content
    assert len(state["earlier_messages"]) == 4
    assert all(len(m["text"]) <= 600 for m in state["earlier_messages"])
    assert state["earlier_messages"][-1]["speaker"] == "companion"
    assert state["conversation_notes"] == {"topic": "work"}


async def test_sdk_debug_logging_never_records_bodies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.observability import configure_logging

    configure_logging("DEBUG")
    root = logging.getLogger()
    root.addHandler(caplog.handler)
    logging.getLogger("typesafe_sdk").setLevel(logging.DEBUG)  # as TYPESAFE_LOG_LEVEL=debug would
    try:
        backend = make_backend(Recorder(ok(valid_answers())))  # caps the SDK logger again
        await backend.ask({"latest_message_from_person": "SENTINEL-BODY"}, list(QUESTIONS))
        logging.getLogger("app.test").debug("sentinel-visible")
    finally:
        root.removeHandler(caplog.handler)
        configure_logging("INFO")
    assert "sentinel-visible" in caplog.text  # the capture works
    assert "SENTINEL-BODY" not in caplog.text

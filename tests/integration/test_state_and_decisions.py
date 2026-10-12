"""Phase 2: conversation state, decisions and the response policy, end to end.

Uses PostgreSQL, the scripted LLM, and a scripted decision backend.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import ConversationStateRecord, Message, TurnDecision
from app.decisions import thresholds as thresholds_module
from app.decisions.backend import DecisionBackendError
from app.decisions.questions import CONVERSATION_MODE, QUESTIONS, RESPONSE_STRATEGY
from app.decisions.schemas import FallbackReason
from app.llm.schemas import LLMError, LLMErrorCategory
from app.prompting import PromptRegistry
from tests.helpers import (
    ALICE,
    BOB,
    ScriptedDecisionBackend,
    ScriptedLLMClient,
    blocking_step,
    choice,
    create_conversation,
    make_settings,
    running_app,
)

PROMPT = PromptRegistry().get("companion_system", "1.1.0")


def extraction(topic: str, goal: str = "", provenance: str = "unknown") -> str:
    return json.dumps(
        {
            "topic": topic,
            "subtopics": [],
            "user_goal": {"value": goal, "provenance": provenance},
            "emotional_context": {"value": "", "provenance": "unknown"},
            "open_questions": [],
            "recently_discussed": [topic],
        }
    )


def answers(mode: str = "listening", strategy: str = "validate") -> dict[str, Any]:
    return {
        "conversation_mode": choice(CONVERSATION_MODE, mode),
        "response_strategy": choice(RESPONSE_STRATEGY, strategy),
        "knowledge_retrieval_useful": {"type": "noul", "noul": 0.2},
    }


@pytest.fixture
def backend() -> ScriptedDecisionBackend:
    return ScriptedDecisionBackend(answers())


@pytest.fixture
async def laya_client(
    database_url: str,
    clean_db: None,
    llm: ScriptedLLMClient,
    backend: ScriptedDecisionBackend,
) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(
        database_url=database_url, decision_provider="laya", laya_api_key="laya-test"
    )
    async with running_app(settings, llm, backend) as (_, client):
        yield client


@pytest.fixture
def calibrated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend an evaluation justified acting on both choice questions."""
    monkeypatch.setitem(
        thresholds_module.THRESHOLDS,
        "laya",
        {CONVERSATION_MODE.key: 0.5, RESPONSE_STRATEGY.key: 0.5},
    )


async def send(client: httpx.AsyncClient, cid: str, content: str) -> httpx.Response:
    response = await client.post(
        f"/v1/conversations/{cid}/messages", json={"content": content}, headers=ALICE
    )
    assert response.status_code == 201, response.text
    return response


async def stored_state(db_engine: AsyncEngine, cid: str) -> ConversationStateRecord | None:
    async with db_engine.connect() as conn:
        row = (
            await conn.execute(
                select(ConversationStateRecord).where(
                    ConversationStateRecord.conversation_id == uuid.UUID(cid)
                )
            )
        ).one_or_none()
    return row  # type: ignore[return-value]


async def test_state_is_extracted_persisted_and_used_next_turn(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient, db_engine: AsyncEngine
) -> None:
    llm.queue_state(
        extraction("job interview", "prepare for Friday's interview", "stated"),
        extraction("job interview nerves", "prepare for Friday's interview", "stated"),
    )
    cid = await create_conversation(laya_client, ALICE)
    await send(laya_client, cid, "I have a job interview on Friday and want to prepare.")
    first = await stored_state(db_engine, cid)
    assert first is not None and first.state_version == 1
    assert first.state["topic"] == "job interview"
    assert first.state["user_goal"] == {
        "value": "prepare for Friday's interview",
        "provenance": "stated",
    }
    assert first.extraction is not None
    assert first.extraction["prompt"] == "state_extraction@1.0.0"

    await send(laya_client, cid, "I'm really nervous about it.")
    second = await stored_state(db_engine, cid)
    assert second is not None and second.state_version == 2
    assert second.state["topic"] == "job interview nerves"

    # The second extraction saw the previous notes as data in a user message.
    state_request = llm.state_requests[1]
    assert state_request.response_schema is not None
    assert "job interview" in state_request.messages[0].content
    assert "<previous_notes>" in state_request.messages[0].content
    # The second reply carried the notes in the user role, never in the system prompt.
    reply_request = llm.requests[1]
    assert "<conversation_notes>" in reply_request.messages[0].content
    assert "job interview" not in reply_request.system


async def test_extraction_failure_keeps_previous_state_and_the_turn_completes(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient, db_engine: AsyncEngine
) -> None:
    llm.queue_state(
        extraction("moving house"),
        LLMError(LLMErrorCategory.RATE_LIMITED, "rate limited"),
        "not json at all",
    )
    cid = await create_conversation(laya_client, ALICE)
    await send(laya_client, cid, "We're moving house next month.")
    await send(laya_client, cid, "Packing is endless.")
    await send(laya_client, cid, "Anyway.")
    record = await stored_state(db_engine, cid)
    assert record is not None and record.state_version == 1
    assert record.state["topic"] == "moving house"
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                select(Message.generation_metadata)
                .where(Message.conversation_id == uuid.UUID(cid), Message.role == "assistant")
                .order_by(Message.sequence)
            )
        ).scalars()
        failures = [m["state"]["failure"] for m in rows]
    assert failures == [None, "llm:rate_limited", "invalid_output"]


async def test_decisions_are_stored_with_each_completed_turn(
    laya_client: httpx.AsyncClient, backend: ScriptedDecisionBackend, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(laya_client, ALICE)
    response = await send(laya_client, cid, "Work has been brutal this week.")
    user_message_id = uuid.UUID(response.json()["user_message_id"])
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                select(TurnDecision).where(TurnDecision.user_message_id == user_message_id)
            )
        ).all()
    by_question = {r.question_id: r for r in rows}
    assert set(by_question) == {q.id for q in QUESTIONS}
    strategy = by_question["response_strategy"]
    # No thresholds yet: shadow mode stores the answer but uses the fallback.
    assert strategy.returned_value == "validate"
    assert strategy.value == RESPONSE_STRATEGY.fallback
    assert strategy.fallback and strategy.fallback_reason == "not_calibrated"
    assert strategy.provider == "laya" and strategy.model_id == "laya-test"
    assert strategy.question_version == "1.0.0"
    # The decision state was built from the latest message.
    assert backend.calls[0]["latest_message_from_person"] == "Work has been brutal this week."


async def test_uncalibrated_decisions_leave_the_reply_prompt_unconstrained(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(laya_client, ALICE)
    await send(laya_client, cid, "Work has been brutal this week.")
    assert PROMPT.fragment("open") in llm.requests[-1].system
    assert PROMPT.fragment("validate") not in llm.requests[-1].system


@pytest.mark.usefixtures("calibrated")
async def test_calibrated_strategy_decision_shapes_the_reply_prompt(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(laya_client, ALICE)
    response = await send(laya_client, cid, "Work has been brutal this week.")
    system = llm.requests[-1].system
    assert PROMPT.fragment("validate") in system
    assert PROMPT.fragment("open") not in system
    async with db_engine.connect() as conn:
        metadata = (
            await conn.execute(
                select(Message.generation_metadata).where(
                    Message.reply_to_message_id == uuid.UUID(response.json()["user_message_id"])
                )
            )
        ).scalar_one()
    assert metadata["policy"]["source"] == "decision"
    assert metadata["policy"]["strategy"] == "validate"
    record = await stored_state(db_engine, cid)
    assert record is not None and record.state["conversation_mode"] == "listening"


@pytest.mark.usefixtures("calibrated")
async def test_explicit_request_overrides_the_decision(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient, backend: ScriptedDecisionBackend
) -> None:
    backend.answers = answers(mode="problem_solving", strategy="problem_solve")
    cid = await create_conversation(laya_client, ALICE)
    await send(laya_client, cid, "I don't want advice, I just need to vent about my boss.")
    system = llm.requests[-1].system
    assert PROMPT.fragment("no_advice") in system
    assert PROMPT.fragment("validate") in system
    assert PROMPT.fragment("problem_solve") not in system


@pytest.mark.usefixtures("calibrated")
async def test_safety_mode_signal_takes_precedence(
    laya_client: httpx.AsyncClient, llm: ScriptedLLMClient, backend: ScriptedDecisionBackend
) -> None:
    backend.answers = answers(mode="safety", strategy="plan")
    cid = await create_conversation(laya_client, ALICE)
    await send(laya_client, cid, "Help me plan my week, nothing matters anyway.")
    system = llm.requests[-1].system
    assert PROMPT.fragment("safety_signal") in system
    assert PROMPT.fragment("plan") not in system


@pytest.mark.usefixtures("calibrated")
async def test_decision_outage_falls_back_and_the_turn_completes(
    laya_client: httpx.AsyncClient,
    llm: ScriptedLLMClient,
    backend: ScriptedDecisionBackend,
    db_engine: AsyncEngine,
) -> None:
    backend.error = DecisionBackendError(FallbackReason.UNAVAILABLE, "down")
    cid = await create_conversation(laya_client, ALICE)
    response = await send(laya_client, cid, "Work has been brutal this week.")
    assert PROMPT.fragment("open") in llm.requests[-1].system
    async with db_engine.connect() as conn:
        reasons = (
            await conn.execute(
                select(TurnDecision.fallback_reason).where(
                    TurnDecision.user_message_id == uuid.UUID(response.json()["user_message_id"])
                )
            )
        ).scalars()
        assert set(reasons) == {"unavailable"}


async def test_decisions_never_run_for_a_foreign_conversation(
    laya_client: httpx.AsyncClient, backend: ScriptedDecisionBackend, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(laya_client, ALICE)
    response = await laya_client.post(
        f"/v1/conversations/{cid}/messages", json={"content": "hi"}, headers=BOB
    )
    assert response.status_code == 404
    assert backend.calls == [] and llm.state_requests == [] and llm.requests == []


async def test_stale_state_write_is_rejected_but_the_reply_is_kept(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    llm = ScriptedLLMClient()
    llm.queue_state(extraction("first topic"), extraction("from the slow turn"))
    release, entered = asyncio.Event(), asyncio.Event()
    settings = make_settings(database_url=database_url)
    async with running_app(settings, llm) as (_, client):
        cid = await create_conversation(client, ALICE)
        await send(client, cid, "first")  # state version 1
        llm.queue(blocking_step(release, entered))
        slow = asyncio.create_task(send(client, cid, "second"))
        await entered.wait()  # turn 2 read version 1 and is generating
        async with db_engine.begin() as conn:  # a concurrent writer moves the state on
            await conn.execute(
                update(ConversationStateRecord)
                .where(ConversationStateRecord.conversation_id == uuid.UUID(cid))
                .values(state_version=2, state=text('\'{"topic": "newer"}\'::jsonb'))
            )
        release.set()
        response = await slow
    assert response.json()["assistant_content"] == "reply to: second"
    record = await stored_state(db_engine, cid)
    assert record is not None and record.state_version == 2
    assert record.state == {"topic": "newer"}


async def test_state_extraction_can_be_disabled(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    llm = ScriptedLLMClient()
    settings = make_settings(database_url=database_url, state_extraction_enabled=False)
    async with running_app(settings, llm) as (app, client):
        cid = await create_conversation(client, ALICE)
        await send(client, cid, "hello")
        assert isinstance(app, FastAPI)
    assert llm.state_requests == []
    assert await stored_state(db_engine, cid) is None

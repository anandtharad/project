"""Failure matrix (spec §23): every dependency failure has a defined, tested outcome.

Common invariants: the user message is preserved, no assistant message is
fabricated, and the response uses the error contract.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Message
from app.llm.schemas import FinishReason, GenerationRequest, LLMError, LLMErrorCategory
from tests.helpers import (
    ALICE,
    ScriptedLLMClient,
    create_conversation,
    list_messages,
    make_response,
    make_settings,
    running_app,
    send,
)


async def failure_category(engine: AsyncEngine, user_message_id: str | None = None) -> str | None:
    async with engine.connect() as conn:
        stmt = select(Message.failure_category).where(Message.role == "user")
        return await conn.scalar(stmt.order_by(Message.created_at.desc()))


async def assert_user_message_preserved_without_reply(
    client: httpx.AsyncClient, cid: str, content: str, status: str = "failed"
) -> None:
    messages = await list_messages(client, cid)
    assert [(m["role"], m["content"], m["status"]) for m in messages] == [("user", content, status)]


async def test_turn_deadline_exceeded(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    llm = ScriptedLLMClient()

    async def hang(_: GenerationRequest) -> str:
        await asyncio.sleep(10)
        return "too late"

    llm.queue(hang)
    settings = make_settings(
        database_url=database_url, turn_deadline_seconds=0.3, stale_turn_seconds=5
    )
    async with running_app(settings, llm) as (_, client):
        cid = await create_conversation(client, ALICE)
        response = await send(client, cid, "Slow one")
        assert response.status_code == 504
        error = response.json()["error"]
        assert error["category"] == "timeout"
        assert error["code"] == "generation_timeout"
        await assert_user_message_preserved_without_reply(client, cid, "Slow one")
    assert await failure_category(db_engine) == "timeout"


@pytest.mark.parametrize(
    ("error", "retry_after_header", "stored_category"),
    [
        (
            LLMError(LLMErrorCategory.RATE_LIMITED, "slow down", retry_after_seconds=3),
            "4",
            "llm:rate_limited",
        ),
        (LLMError(LLMErrorCategory.TIMEOUT, "provider timeout"), None, "llm:timeout"),
        (LLMError(LLMErrorCategory.CONNECTION, "no route"), None, "llm:connection"),
        (LLMError(LLMErrorCategory.AUTHENTICATION, "bad key"), None, "llm:authentication"),
        (LLMError(LLMErrorCategory.INVALID_REQUEST, "bad"), None, "llm:invalid_request"),
    ],
)
async def test_provider_errors_are_controlled(
    client: httpx.AsyncClient,
    llm: ScriptedLLMClient,
    db_engine: AsyncEngine,
    error: LLMError,
    retry_after_header: str | None,
    stored_category: str,
) -> None:
    llm.queue(error)
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Hi")
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["category"] == "dependency_unavailable"
    assert body["code"] == "generation_unavailable"
    assert response.headers.get("retry-after") == retry_after_header
    assert error.message not in response.text  # provider detail is not exposed
    await assert_user_message_preserved_without_reply(client, cid, "Hi")
    assert await failure_category(db_engine) == stored_category


@pytest.mark.parametrize(
    ("generated", "reason"),
    [
        (make_response("", finish_reason=FinishReason.REFUSAL), "refused"),
        (make_response("Half a sent", finish_reason=FinishReason.MAX_TOKENS), "truncated"),
        (make_response("   "), "empty"),
        (make_response("x" * 9000), "too_long"),
    ],
)
async def test_invalid_generations_are_not_persisted(
    client: httpx.AsyncClient,
    llm: ScriptedLLMClient,
    db_engine: AsyncEngine,
    generated: Any,
    reason: str,
) -> None:
    llm.queue(generated)
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Hi")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "invalid_generation"
    await assert_user_message_preserved_without_reply(client, cid, "Hi")
    assert await failure_category(db_engine) == f"invalid_response:{reason}"


async def test_persistence_failure_is_not_reported_as_complete(
    app: FastAPI,
    client: httpx.AsyncClient,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken_complete(**_: Any) -> None:
        raise OperationalError("INSERT", {}, ConnectionError("connection lost"))

    monkeypatch.setattr(app.state.orchestrator._store, "complete", broken_complete)
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Save me")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "persistence_failed"
    assert "assistant_content" not in response.text
    await assert_user_message_preserved_without_reply(client, cid, "Save me")
    assert await failure_category(db_engine) == "persistence_error"


async def test_double_database_failure_leaves_turn_recoverable(
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If marking the turn failed also fails, the turn stays pending and is
    recovered by stale-turn takeover (tested in test_concurrency)."""

    async def broken(**_: Any) -> None:
        raise OperationalError("UPDATE", {}, ConnectionError("connection lost"))

    store = app.state.orchestrator._store
    monkeypatch.setattr(store, "complete", broken)
    monkeypatch.setattr(store, "fail", broken)
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Save me")
    assert response.status_code == 503
    await assert_user_message_preserved_without_reply(client, cid, "Save me", status="pending")


@pytest.fixture
async def unhandled_error_client(
    database_url: str, clean_db: None
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    async with running_app(make_settings(database_url=database_url), ScriptedLLMClient()) as pair:
        yield pair


async def test_unexpected_errors_return_generic_500(
    unhandled_error_client: tuple[FastAPI, httpx.AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, client = unhandled_error_client

    async def explode(**_: Any) -> None:
        raise RuntimeError("internal detail sk-ant-secret /etc/passwd")

    monkeypatch.setattr(app.state.orchestrator, "submit_message", explode)
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Hi")
    assert response.status_code == 500
    assert response.json()["error"] == {
        "category": "internal_error",
        "code": "internal_error",
        "message": "An internal error occurred.",
        "trace_id": response.headers["x-request-id"],
    }
    for leaked in ("sk-ant-secret", "Traceback", "RuntimeError", "/etc/passwd"):
        assert leaked not in response.text


async def test_unknown_conversation_is_rejected_before_model_call(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    response = await send(client, str(uuid.uuid4()), "Hello?")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"
    assert llm.requests == []

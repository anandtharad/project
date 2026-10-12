"""Concurrent requests: one in-flight turn per conversation, gap-free ordering."""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Message
from app.orchestration.persistence import AssistantProvenance, TurnAttemptLost
from tests.helpers import (
    ALICE,
    ScriptedLLMClient,
    blocking_step,
    create_conversation,
    list_messages,
    send,
)


async def test_second_message_while_turn_in_flight_is_rejected(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    release, entered = asyncio.Event(), asyncio.Event()
    llm.queue(blocking_step(release, entered))
    cid = await create_conversation(client, ALICE)

    first = asyncio.create_task(send(client, cid, "first"))
    await asyncio.wait_for(entered.wait(), 5)
    second = await send(client, cid, "second")
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "turn_in_progress"
    assert second.headers["retry-after"] == "2"

    release.set()
    assert (await first).status_code == 201
    messages = await list_messages(client, cid)
    assert [(m["sequence"], m["content"]) for m in messages] == [
        (1, "first"),
        (2, "reply to: first"),
    ]


async def test_duplicate_in_flight_request_does_not_duplicate_turn(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    release, entered = asyncio.Event(), asyncio.Event()
    llm.queue(blocking_step(release, entered))
    cid = await create_conversation(client, ALICE)

    first = asyncio.create_task(send(client, cid, "hello", idempotency_key="dup"))
    await asyncio.wait_for(entered.wait(), 5)
    in_flight = await send(client, cid, "hello", idempotency_key="dup")
    assert in_flight.status_code == 409
    assert in_flight.json()["error"]["code"] == "turn_in_progress"

    release.set()
    completed = await first
    replay = await send(client, cid, "hello", idempotency_key="dup")
    assert replay.status_code == 200
    assert replay.json()["assistant_message_id"] == completed.json()["assistant_message_id"]
    assert len(llm.requests) == 1
    assert len(await list_messages(client, cid)) == 2


async def test_burst_of_parallel_requests_yields_exactly_one_turn(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    release = asyncio.Event()
    llm.queue(blocking_step(release))
    cid = await create_conversation(client, ALICE)

    tasks = [asyncio.create_task(send(client, cid, f"msg {i}")) for i in range(8)]
    # Wait until every request except the winner has been answered.
    for _ in range(100):
        if sum(t.done() for t in tasks) == 7:
            break
        await asyncio.sleep(0.05)
    release.set()
    statuses = sorted(r.status_code for r in await asyncio.gather(*tasks))
    assert statuses == [201] + [409] * 7
    messages = await list_messages(client, cid)
    assert [m["sequence"] for m in messages] == [1, 2]


async def test_different_conversations_run_in_parallel(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    release = asyncio.Event()
    entered_a, entered_b = asyncio.Event(), asyncio.Event()
    llm.queue(blocking_step(release, entered_a), blocking_step(release, entered_b))
    a = await create_conversation(client, ALICE)
    b = await create_conversation(client, ALICE)

    task_a = asyncio.create_task(send(client, a, "in a"))
    task_b = asyncio.create_task(send(client, b, "in b"))
    await asyncio.wait_for(asyncio.gather(entered_a.wait(), entered_b.wait()), 5)
    release.set()
    assert (await task_a).status_code == 201
    assert (await task_b).status_code == 201


async def test_abandoned_pending_turn_is_taken_over(
    app: FastAPI, client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    # Simulate a crashed worker: a pending turn whose attempt started long ago.
    abandoned_id, abandoned_attempt = uuid.uuid4(), uuid.uuid4()
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO messages (id, conversation_id, sequence, role, content, turn_status,"
                " attempt_id, attempt_started_at, request_fingerprint)"
                " VALUES (:id, :cid, 1, 'user', 'lost message', 'pending', :attempt,"
                " now() - interval '1 hour', 'x')"
            ),
            {"id": abandoned_id, "cid": uuid.UUID(cid), "attempt": abandoned_attempt},
        )
        await conn.execute(
            text("UPDATE conversations SET last_sequence = 1 WHERE id = :cid"),
            {"cid": uuid.UUID(cid)},
        )

    response = await send(client, cid, "new message")
    assert response.status_code == 201
    messages = await list_messages(client, cid)
    assert [(m["sequence"], m["status"]) for m in messages] == [
        (1, "failed"),
        (2, "completed"),
        (3, None),
    ]

    # The abandoned attempt can no longer complete its turn.
    with pytest.raises(TurnAttemptLost):
        await app.state.orchestrator._store.complete(
            conversation_id=uuid.UUID(cid),
            user_message_id=abandoned_id,
            attempt_id=abandoned_attempt,
            content="late reply",
            provenance=AssistantProvenance(
                provider="x", model_id="x", prompt_id="x", prompt_version="x"
            ),
        )
    async with db_engine.connect() as conn:
        failure = await conn.scalar(
            select(Message.failure_category).where(Message.id == abandoned_id)
        )
    assert failure == "abandoned"


async def test_database_enforces_single_pending_turn(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = uuid.UUID(await create_conversation(client, ALICE))
    insert = text(
        "INSERT INTO messages (id, conversation_id, sequence, role, content, turn_status)"
        " VALUES (:id, :cid, :seq, 'user', 'x', 'pending')"
    )
    async with db_engine.begin() as conn:
        await conn.execute(insert, {"id": uuid.uuid4(), "cid": cid, "seq": 1})
    with pytest.raises(IntegrityError):
        async with db_engine.begin() as conn:
            await conn.execute(insert, {"id": uuid.uuid4(), "cid": cid, "seq": 2})


async def test_database_enforces_one_reply_per_user_message(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "hello")
    user_message_id = uuid.UUID(response.json()["user_message_id"])
    with pytest.raises(IntegrityError):
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO messages (id, conversation_id, sequence, role, content,"
                    " reply_to_message_id, model_id, prompt_id, prompt_version)"
                    " VALUES (:id, :cid, 3, 'assistant', 'dup', :reply, 'm', 'p', 'v')"
                ),
                {"id": uuid.uuid4(), "cid": uuid.UUID(cid), "reply": user_message_id},
            )

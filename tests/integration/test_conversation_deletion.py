"""DELETE /v1/conversations/{id}: removes the conversation and everything
stored per turn; keeps memories; refuses while a reply is in flight; never
touches or reveals another user's conversation."""

from __future__ import annotations

import asyncio
import uuid

import httpx
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.helpers import (
    ALICE,
    BOB,
    ScriptedLLMClient,
    blocking_step,
    create_conversation,
    send,
)


async def _count(db_engine: AsyncEngine, table: str, cid: str) -> int:
    async with db_engine.connect() as conn:
        value = await conn.scalar(
            text(f"SELECT count(*) FROM {table} WHERE conversation_id = :cid"),  # noqa: S608
            {"cid": uuid.UUID(cid)},
        )
    return int(value or 0)


async def test_owner_deletes_conversation_and_its_turns(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    assert (await send(client, cid, "hello there")).status_code == 201
    assert await _count(db_engine, "messages", cid) == 2

    response = await client.delete(f"/v1/conversations/{cid}", headers=ALICE)
    assert response.status_code == 204
    assert (await client.get(f"/v1/conversations/{cid}", headers=ALICE)).status_code == 404
    assert await _count(db_engine, "messages", cid) == 0
    assert await _count(db_engine, "turn_decisions", cid) == 0
    # Deleting again is a plain 404, not an error.
    again = await client.delete(f"/v1/conversations/{cid}", headers=ALICE)
    assert again.status_code == 404


async def test_memories_survive_with_their_conversation_link_cleared(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    async with db_engine.begin() as conn:
        user_id = await conn.scalar(
            text("SELECT user_id FROM conversations WHERE id = :cid"), {"cid": uuid.UUID(cid)}
        )
        memory_id = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO memories (id, user_id, source_conversation_id, content, memory_type,"
                " status, importance, provenance, schema_version)"
                " VALUES (:id, :uid, :cid, 'likes tea', 'preference', 'active', 2, '{}', 1)"
            ),
            {"id": memory_id, "uid": user_id, "cid": uuid.UUID(cid)},
        )

    assert (await client.delete(f"/v1/conversations/{cid}", headers=ALICE)).status_code == 204
    async with db_engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT source_conversation_id FROM memories WHERE id = :id"),
                {"id": memory_id},
            )
        ).one()
    assert row.source_conversation_id is None


async def test_other_users_conversation_cannot_be_deleted_or_probed(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    assert (await send(client, cid, "mine")).status_code == 201

    foreign = await client.delete(f"/v1/conversations/{cid}", headers=BOB)
    missing = await client.delete(f"/v1/conversations/{uuid.uuid4()}", headers=BOB)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json()["error"]["code"] == missing.json()["error"]["code"]
    assert await _count(db_engine, "messages", cid) == 2


async def test_delete_is_refused_while_a_reply_is_in_flight(
    client: httpx.AsyncClient, llm: ScriptedLLMClient, db_engine: AsyncEngine
) -> None:
    release, entered = asyncio.Event(), asyncio.Event()
    llm.queue(blocking_step(release, entered))
    cid = await create_conversation(client, ALICE)

    turn = asyncio.create_task(send(client, cid, "first"))
    await asyncio.wait_for(entered.wait(), 5)
    refused = await client.delete(f"/v1/conversations/{cid}", headers=ALICE)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "turn_in_progress"

    release.set()
    assert (await turn).status_code == 201
    assert (await client.delete(f"/v1/conversations/{cid}", headers=ALICE)).status_code == 204
    assert await _count(db_engine, "messages", cid) == 0


async def test_an_abandoned_turn_does_not_block_deletion(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO messages (id, conversation_id, sequence, role, content, turn_status,"
                " attempt_id, attempt_started_at, request_fingerprint)"
                " VALUES (:id, :cid, 1, 'user', 'lost', 'pending', :attempt,"
                " now() - interval '1 hour', 'x')"
            ),
            {"id": uuid.uuid4(), "cid": uuid.UUID(cid), "attempt": uuid.uuid4()},
        )
    assert (await client.delete(f"/v1/conversations/{cid}", headers=ALICE)).status_code == 204


async def test_delete_requires_credentials(client: httpx.AsyncClient) -> None:
    cid = await create_conversation(client, ALICE)
    response = await client.delete(f"/v1/conversations/{cid}")
    assert response.status_code == 401


async def test_retention_purges_only_inactive_conversations(
    app: FastAPI, client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    """CONVERSATION_RETENTION_DAYS: a conversation with no new message for
    that long goes with its turns; a recent one stays; memories stay."""
    old = await create_conversation(client, ALICE)
    recent = await create_conversation(client, ALICE)
    for cid in (old, recent):
        assert (await send(client, cid, "hello there")).status_code == 201
    async with db_engine.begin() as conn:
        user_id = await conn.scalar(
            text("SELECT user_id FROM conversations WHERE id = :cid"), {"cid": uuid.UUID(old)}
        )
        await conn.execute(
            text(
                "UPDATE conversations SET updated_at = now() - interval '31 days' WHERE id = :cid"
            ),
            {"cid": uuid.UUID(old)},
        )
        await conn.execute(
            text(
                "INSERT INTO memories (id, user_id, source_conversation_id, content, memory_type,"
                " status, importance, provenance, schema_version)"
                " VALUES (:id, :uid, :cid, 'likes tea', 'preference', 'active', 2, '{}', 1)"
            ),
            {"id": uuid.uuid4(), "uid": user_id, "cid": uuid.UUID(old)},
        )

    assert await app.state.turn_store.purge_inactive_conversations(30) == 1
    assert (await client.get(f"/v1/conversations/{old}", headers=ALICE)).status_code == 404
    assert await _count(db_engine, "messages", old) == 0
    assert await _count(db_engine, "turn_decisions", old) == 0
    assert await _count(db_engine, "messages", recent) == 2
    async with db_engine.connect() as conn:
        kept = await conn.scalar(
            text("SELECT count(*) FROM memories WHERE user_id = :uid"), {"uid": user_id}
        )
    assert kept == 1
    # Nothing left to purge.
    assert await app.state.turn_store.purge_inactive_conversations(30) == 0

"""Idempotency is enforced in PostgreSQL, not process memory."""

from __future__ import annotations

import httpx

from app.llm.schemas import LLMError, LLMErrorCategory
from tests.helpers import ALICE, ScriptedLLMClient, create_conversation, list_messages, send


async def test_duplicate_request_replays_completed_turn(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(client, ALICE)
    first = await send(client, cid, "Hello", idempotency_key="key-1")
    second = await send(client, cid, "Hello", idempotency_key="key-1")

    assert first.status_code == 201
    assert second.status_code == 200
    a, b = first.json(), second.json()
    for field in ("user_message_id", "assistant_message_id", "assistant_content", "created_at"):
        assert a[field] == b[field]
    assert b["replayed"] is True
    assert len(llm.requests) == 1
    assert len(await list_messages(client, cid)) == 2


async def test_key_reused_for_different_content_is_rejected(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(client, ALICE)
    await send(client, cid, "Hello", idempotency_key="key-1")
    conflict = await send(client, cid, "Something else", idempotency_key="key-1")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_reused"
    assert len(llm.requests) == 1
    assert len(await list_messages(client, cid)) == 2


async def test_requests_without_key_are_separate_turns(client: httpx.AsyncClient) -> None:
    cid = await create_conversation(client, ALICE)
    first = await send(client, cid, "Hello")
    second = await send(client, cid, "Hello")
    assert first.json()["user_message_id"] != second.json()["user_message_id"]
    assert len(await list_messages(client, cid)) == 4


async def test_keys_are_scoped_to_the_conversation(client: httpx.AsyncClient) -> None:
    first = await create_conversation(client, ALICE)
    second = await create_conversation(client, ALICE)
    a = await send(client, first, "Hello", idempotency_key="shared")
    b = await send(client, second, "Hello", idempotency_key="shared")
    assert a.status_code == b.status_code == 201
    assert a.json()["user_message_id"] != b.json()["user_message_id"]


async def test_failed_turn_retried_with_same_key_reuses_user_message(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    llm.queue(LLMError(LLMErrorCategory.OVERLOADED, "busy"), "Recovered reply")
    cid = await create_conversation(client, ALICE)

    failed = await send(client, cid, "Are you there?", idempotency_key="retry-me")
    assert failed.status_code == 503
    after_failure = await list_messages(client, cid)
    assert [(m["role"], m["status"]) for m in after_failure] == [("user", "failed")]

    retried = await send(client, cid, "Are you there?", idempotency_key="retry-me")
    assert retried.status_code == 201
    assert retried.json()["assistant_content"] == "Recovered reply"
    messages = await list_messages(client, cid)
    assert [(m["sequence"], m["role"], m["status"]) for m in messages] == [
        (1, "user", "completed"),
        (2, "assistant", None),
    ]
    assert retried.json()["user_message_id"] == messages[0]["message_id"]
    assert len(llm.requests) == 2


async def test_failed_turn_cannot_be_retried_after_conversation_moves_on(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    llm.queue(LLMError(LLMErrorCategory.OVERLOADED, "busy"))
    cid = await create_conversation(client, ALICE)
    assert (await send(client, cid, "first", idempotency_key="k1")).status_code == 503
    assert (await send(client, cid, "second", idempotency_key="k2")).status_code == 201

    stale_retry = await send(client, cid, "first", idempotency_key="k1")
    assert stale_retry.status_code == 409
    assert stale_retry.json()["error"]["code"] == "turn_superseded"


async def test_failed_user_message_stays_in_later_context(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    llm.queue(LLMError(LLMErrorCategory.OVERLOADED, "busy"))
    cid = await create_conversation(client, ALICE)
    await send(client, cid, "My sister is visiting next week.")
    await send(client, cid, "I'm nervous about it.")
    # Adjacent user messages are merged so the model sees both, in order.
    assert [m.content for m in llm.requests[-1].messages] == [
        "My sister is visiting next week.\n\nI'm nervous about it."
    ]

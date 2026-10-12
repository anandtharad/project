"""End-to-end text conversation against PostgreSQL with a scripted LLM."""

from __future__ import annotations

import uuid

import httpx
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Message
from app.llm.schemas import ChatRole, GenerationPurpose
from app.prompting import PromptRegistry
from tests.helpers import (
    ALICE,
    ScriptedLLMClient,
    create_conversation,
    default_system_prompt,
    list_messages,
    send,
)


async def test_create_and_get_conversation(client: httpx.AsyncClient) -> None:
    created = await client.post("/v1/conversations", headers=ALICE)
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "active"
    uuid.UUID(body["conversation_id"])

    detail = await client.get(f"/v1/conversations/{body['conversation_id']}", headers=ALICE)
    assert detail.status_code == 200
    assert detail.json()["message_count"] == 0


async def test_multi_turn_context_is_preserved_in_order(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    llm.queue(
        "What's been feeling most overwhelming?",
        "That sounds frustrating. Are the expectations changing, or not communicated clearly?",
        "So you're held accountable for expectations that don't feel stable.",
    )
    cid = await create_conversation(client, ALICE)
    user_turns = [
        "I've been feeling overwhelmed lately.",
        "Work. My manager keeps changing what he expects from me.",
        "I think it's both. And then I get blamed for not delivering.",
    ]
    replies = []
    for text in user_turns:
        response = await send(client, cid, text)
        assert response.status_code == 201, response.text
        replies.append(response.json()["assistant_content"])

    # The third request carried the whole conversation, in order.
    third = llm.requests[2]
    assert [(m.role, m.content) for m in third.messages] == [
        (ChatRole.USER, user_turns[0]),
        (ChatRole.ASSISTANT, replies[0]),
        (ChatRole.USER, user_turns[1]),
        (ChatRole.ASSISTANT, replies[1]),
        (ChatRole.USER, user_turns[2]),
    ]
    # The system prompt is the versioned prompt, separate from user content.
    expected_system = default_system_prompt()
    assert all(r.system == expected_system for r in llm.requests)
    assert all(r.trace.purpose is GenerationPurpose.CONVERSATION for r in llm.requests)

    messages = await list_messages(client, cid)
    assert [m["sequence"] for m in messages] == [1, 2, 3, 4, 5, 6]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 3
    assert [m["content"] for m in messages[1::2]] == replies
    assert all(m["status"] == "completed" for m in messages[0::2])
    assert all(m["status"] is None for m in messages[1::2])

    detail = await client.get(f"/v1/conversations/{cid}", headers=ALICE)
    assert detail.json()["message_count"] == 6


async def test_response_payload_contract(client: httpx.AsyncClient) -> None:
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Hello there")
    body = response.json()
    assert set(body) == {
        "conversation_id",
        "user_message_id",
        "assistant_message_id",
        "assistant_content",
        "created_at",
        "replayed",
        "timing_ms",
    }
    assert body["conversation_id"] == cid
    assert body["assistant_content"] == "reply to: Hello there"
    assert body["replayed"] is False
    assert "completed" in body["timing_ms"]


async def test_generation_request_uses_configured_settings(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    settings = app.state.settings
    resolved = app.state.resolved_provider
    cid = await create_conversation(client, ALICE)
    await send(client, cid, "hi")
    request = llm.requests[-1]
    assert request.model == resolved.model == "fake-stub"
    assert request.max_output_tokens == resolved.max_output_tokens
    assert request.timeout_seconds == settings.generation_timeout_seconds
    assert request.params.effort == resolved.effort
    assert request.trace.conversation_id == cid


async def test_model_and_prompt_versions_are_recorded(
    client: httpx.AsyncClient, db_engine: AsyncEngine
) -> None:
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "Record my provenance")
    assistant_id = uuid.UUID(response.json()["assistant_message_id"])

    prompt = PromptRegistry().get("companion_system", "1.1.0")
    async with db_engine.connect() as conn:
        row = (await conn.execute(select(Message).where(Message.id == assistant_id))).one()
    assert row.provider == "scripted"
    assert row.model_id == "scripted-model"
    assert row.prompt_id == "companion_system"
    assert row.prompt_version == "1.1.0"
    assert row.reply_to_message_id == uuid.UUID(response.json()["user_message_id"])
    metadata = row.generation_metadata
    assert metadata["prompt_checksum"] == prompt.checksum
    assert metadata["requested_model"] == "scripted-model"
    assert metadata["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }
    assert metadata["generation_params"]["max_output_tokens"] > 0
    assert metadata["trace_id"] == response.headers["x-request-id"]
    assert metadata["context"]["included_messages"] == 1


async def test_message_pagination(client: httpx.AsyncClient) -> None:
    cid = await create_conversation(client, ALICE)
    for i in range(3):
        assert (await send(client, cid, f"message {i}")).status_code == 201
    first = await client.get(f"/v1/conversations/{cid}/messages?limit=4", headers=ALICE)
    page = first.json()
    assert [m["sequence"] for m in page["messages"]] == [1, 2, 3, 4]
    assert page["next_after_sequence"] == 4
    second = await client.get(
        f"/v1/conversations/{cid}/messages?limit=4&after_sequence=4", headers=ALICE
    )
    page2 = second.json()
    assert [m["sequence"] for m in page2["messages"]] == [5, 6]
    assert page2["next_after_sequence"] is None


async def test_conversations_are_independent(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    first = await create_conversation(client, ALICE)
    second = await create_conversation(client, ALICE)
    await send(client, first, "about work")
    await send(client, second, "about sleep")
    assert [m.content for m in llm.requests[-1].messages] == ["about sleep"]

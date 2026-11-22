"""The agent's conversation-history tool against the real database: the
person's own messages, in this and their other conversations, and never
anyone else's, whatever the model asks for."""

from __future__ import annotations

import json
import uuid

import httpx
from fastapi import FastAPI
from sqlalchemy import text

from app.agent.tools import ToolContext, execute
from app.llm.schemas import ToolCall
from tests.helpers import ALICE, BOB, ScriptedLLMClient, create_conversation, send


async def user_id(app: FastAPI, external_id: str) -> uuid.UUID:
    async with app.state.session_factory() as session:
        value = await session.scalar(
            text("SELECT id FROM users WHERE external_id = :e"), {"e": external_id}
        )
    assert value is not None
    return uuid.UUID(str(value))


async def search(app: FastAPI, who: uuid.UUID, conversation: str, query: str) -> dict:  # type: ignore[type-arg]
    context = ToolContext(
        user_id=who,
        conversation_id=uuid.UUID(conversation),
        context_from_sequence=10_000,
        state=None,
        memory=None,
        knowledge=None,
        session_factory=app.state.session_factory,
    )
    result = await execute(
        ToolCall(
            id="c1", name="search_conversation_history", arguments=json.dumps({"query": query})
        ),
        context,
        timeout_seconds=5,
        max_characters=4000,
    )
    body = result.content.removeprefix("<conversation_history>\n").removesuffix(
        "\n</conversation_history>"
    )
    return {"status": result.status, **json.loads(body)}


async def test_history_search_is_scoped_to_the_person(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    alice_old = await create_conversation(client, ALICE)
    assert (
        await send(client, alice_old, "My cat Mochi hates the vacuum cleaner")
    ).status_code == 201
    alice_now = await create_conversation(client, ALICE)
    assert (await send(client, alice_now, "Rough week at work")).status_code == 201
    bob = await create_conversation(client, BOB)
    assert (await send(client, bob, "My cat Mochi is my secret", headers=BOB)).status_code == 201
    alice_id = await user_id(app, "dev:alice")
    bob_id = await user_id(app, "dev:bob")

    found = await search(app, alice_id, alice_now, "what did I tell you about my cat")
    texts = [m["text"] for m in found["messages"]]
    assert found["status"] == "ok"
    assert "My cat Mochi hates the vacuum cleaner" in texts
    assert all("secret" not in t for t in texts)  # Bob's message never appears
    assert {m["this_conversation"] for m in found["messages"]} == {False}

    # Bob, even naming Alice's conversation, sees only his own messages.
    bobs = await search(app, bob_id, alice_old, "what did I say about Mochi and the vacuum")
    bob_texts = [m["text"] for m in bobs["messages"]]
    assert "My cat Mochi is my secret" in bob_texts
    assert all("vacuum" not in t for t in bob_texts)  # none of Alice's messages

    none = await search(app, alice_id, alice_now, "spaceship")
    assert none["status"] == "no_matches" and none["messages"] == []


async def test_history_search_leaves_out_safety_turns_from_other_conversations(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    """A crisis disclosure in one chat never resurfaces in another; it stays
    visible inside its own chat."""
    crisis = await create_conversation(client, ALICE)
    response = await send(client, crisis, "i want to kill myself, the exam stress is too much")
    assert response.json()["safety"]["pathway"] != "normal"
    ordinary = await create_conversation(client, ALICE)
    assert (await send(client, ordinary, "the exam stress is wild this week")).status_code == 201
    alice_id = await user_id(app, "dev:alice")

    elsewhere = await search(app, alice_id, ordinary, "exam stress")
    texts = [m["text"] for m in elsewhere["messages"]]
    assert "the exam stress is wild this week" in texts
    assert all("kill myself" not in t for t in texts)
    assert all(
        m["this_conversation"] for m in elsewhere["messages"]
    )  # nothing from the crisis chat

    inside = await search(app, alice_id, crisis, "exam stress")
    assert any("kill myself" in m["text"] for m in inside["messages"])


async def test_history_search_never_returns_deleted_conversations(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    old = await create_conversation(client, ALICE)
    assert (await send(client, old, "my sister Meera is visiting next week")).status_code == 201
    now = await create_conversation(client, ALICE)
    alice_id = await user_id(app, "dev:alice")
    assert (await search(app, alice_id, now, "sister Meera"))["messages"]
    assert (await client.delete(f"/v1/conversations/{old}", headers=ALICE)).status_code == 204
    gone = await search(app, alice_id, now, "sister Meera")
    assert gone["status"] == "no_matches" and gone["messages"] == []


async def test_history_search_is_off_when_memory_is_off(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    old = await create_conversation(client, ALICE)
    assert (await send(client, old, "my sister Meera is visiting next week")).status_code == 201
    now = await create_conversation(client, ALICE)
    alice_id = await user_id(app, "dev:alice")
    context = ToolContext(
        user_id=alice_id,
        conversation_id=uuid.UUID(now),
        context_from_sequence=10_000,
        state=None,
        memory=None,
        knowledge=None,
        session_factory=app.state.session_factory,
        history_enabled=False,
    )
    result = await execute(
        ToolCall(id="c1", name="search_conversation_history", arguments='{"query": "Meera"}'),
        context,
        timeout_seconds=5,
        max_characters=4000,
    )
    assert result.status == "disabled" and "Meera" not in result.content

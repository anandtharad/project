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

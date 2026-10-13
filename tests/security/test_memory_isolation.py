"""Memory access control (spec §10.7, §10.8, Phase 3 task 7).

Every memory query is scoped to the authenticated user in SQL: another
person's memory ID behaves exactly like a missing one, and one person's
memories never reach another person's context.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.helpers import ALICE, BOB, ScriptedLLMClient, create_conversation, send

SECRET = "has a rescue greyhound called Biscuit"


async def alice_remembers(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> dict[str, Any]:
    llm.queue_memory(
        json.dumps(
            {
                "candidates": [
                    {
                        "content": SECRET,
                        "memory_type": "semantic",
                        "explicit": True,
                        "evidence": "a rescue greyhound called Biscuit",
                        "sensitive_category": "none",
                        "lifetime": "long",
                        "importance": 2,
                        "relation": "new",
                        "related_memory": -1,
                    }
                ]
            }
        )
    )
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "I adopted a rescue greyhound called Biscuit last week.")
    assert response.status_code == 201
    await app.state.background.drain()
    listed = await client.get("/v1/users/me/memories", headers=ALICE)
    (memory,) = listed.json()["memories"]
    assert memory["content"] == SECRET
    result: dict[str, Any] = memory
    return result


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/users/me/memories"),
        ("DELETE", "/v1/users/me/memories"),
        ("GET", "/v1/users/me/memories/{mid}"),
        ("DELETE", "/v1/users/me/memories/{mid}"),
        ("POST", "/v1/users/me/memories/{mid}/confirm"),
        ("GET", "/v1/users/me/memory-settings"),
        ("PUT", "/v1/users/me/memory-settings"),
    ],
)
async def test_memory_routes_require_valid_credentials(
    client: httpx.AsyncClient, method: str, path: str
) -> None:
    for headers in ({}, {"Authorization": "Bearer wrong-token"}):
        response = await client.request(
            method,
            path.format(mid=uuid.uuid4()),
            headers=headers,
            json={"memory_enabled": True, "allow_sensitive": False},
        )
        assert response.status_code == 401, (method, path, headers)


async def test_another_users_memory_is_indistinguishable_from_a_missing_one(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    memory = await alice_remembers(app, client, llm)
    mid = memory["memory_id"]
    missing = await client.get(f"/v1/users/me/memories/{uuid.uuid4()}", headers=BOB)
    for method, path in (
        ("GET", f"/v1/users/me/memories/{mid}"),
        ("DELETE", f"/v1/users/me/memories/{mid}"),
        ("POST", f"/v1/users/me/memories/{mid}/confirm"),
    ):
        response = await client.request(method, path, headers=BOB)
        assert response.status_code == 404, (method, path)
        assert response.json()["error"]["code"] == missing.json()["error"]["code"]
    assert (await client.get("/v1/users/me/memories", headers=BOB)).json() == {"memories": []}
    # Bob's "delete all" deletes only Bob's memories.
    assert (await client.delete("/v1/users/me/memories", headers=BOB)).json() == {"deleted": 0}
    still = await client.get(f"/v1/users/me/memories/{mid}", headers=ALICE)
    assert still.status_code == 200 and still.json()["content"] == SECRET


async def test_one_users_memories_never_reach_another_users_context(
    app: FastAPI, client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    await alice_remembers(app, client, llm)
    cid = await create_conversation(client, BOB)
    question = "Any tips for walking a rescue greyhound called Biscuit?"
    response = await send(client, cid, question, headers=BOB)
    assert response.status_code == 201
    await app.state.background.drain()
    assert "<memories>" not in llm.requests[-1].messages[-1].content
    assert "Biscuit" not in llm.requests[-1].system
    # Nor do they reach Bob's extraction call as existing memories.
    assert SECRET not in llm.memory_requests[-1].messages[0].content


async def test_settings_are_per_user(client: httpx.AsyncClient) -> None:
    off = {"memory_enabled": False, "allow_sensitive": False}
    assert (await client.put("/v1/users/me/memory-settings", json=off, headers=ALICE)).json() == off
    bob = await client.get("/v1/users/me/memory-settings", headers=BOB)
    assert bob.json() == {"memory_enabled": True, "allow_sensitive": False}

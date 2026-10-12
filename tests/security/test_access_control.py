"""Authentication and cross-user isolation (FR-019). Ownership is enforced in
application queries; another user's conversation is indistinguishable from a
missing one."""

from __future__ import annotations

import uuid

import httpx
import pytest

from tests.helpers import ALICE, BOB, ScriptedLLMClient, create_conversation, list_messages, send


@pytest.mark.parametrize(
    ("method", "path_template"),
    [
        ("POST", "/v1/conversations"),
        ("GET", "/v1/conversations/{cid}"),
        ("GET", "/v1/conversations/{cid}/messages"),
        ("POST", "/v1/conversations/{cid}/messages"),
    ],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Basic dG9rZW4tYWxpY2U="},
        {"Authorization": "Bearer "},
        {"Authorization": "token-alice"},
    ],
)
async def test_user_data_endpoints_require_valid_credentials(
    client: httpx.AsyncClient, method: str, path_template: str, headers: dict[str, str]
) -> None:
    cid = await create_conversation(client, ALICE)
    response = await client.request(
        method, path_template.format(cid=cid), headers=headers, json={"content": "hi"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["category"] == "unauthorized"


async def test_other_users_conversation_is_indistinguishable_from_missing(
    client: httpx.AsyncClient,
) -> None:
    cid = await create_conversation(client, ALICE)
    foreign = await client.get(f"/v1/conversations/{cid}", headers=BOB)
    missing = await client.get(f"/v1/conversations/{uuid.uuid4()}", headers=BOB)
    assert foreign.status_code == missing.status_code == 404

    def without_trace(response: httpx.Response) -> dict[str, str]:
        return {k: v for k, v in response.json()["error"].items() if k != "trace_id"}

    assert without_trace(foreign) == without_trace(missing)


async def test_other_user_cannot_read_messages(client: httpx.AsyncClient) -> None:
    cid = await create_conversation(client, ALICE)
    await send(client, cid, "private thoughts")
    response = await client.get(f"/v1/conversations/{cid}/messages", headers=BOB)
    assert response.status_code == 404
    assert "private thoughts" not in response.text


async def test_other_user_cannot_post_into_conversation(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(client, ALICE)
    response = await send(client, cid, "injected by bob", headers=BOB)
    assert response.status_code == 404
    assert llm.requests == []
    assert await list_messages(client, cid) == []


async def test_idempotency_key_cannot_replay_another_users_turn(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    cid = await create_conversation(client, ALICE)
    original = await send(client, cid, "my secret", idempotency_key="k-1")
    assert original.status_code == 201
    replay_attempt = await send(client, cid, "my secret", headers=BOB, idempotency_key="k-1")
    assert replay_attempt.status_code == 404
    assert original.json()["assistant_content"] not in replay_attempt.text
    assert len(llm.requests) == 1


async def test_each_user_sees_only_their_own_context(
    client: httpx.AsyncClient, llm: ScriptedLLMClient
) -> None:
    alice_cid = await create_conversation(client, ALICE)
    bob_cid = await create_conversation(client, BOB)
    await send(client, alice_cid, "alice's private detail")
    await send(client, bob_cid, "bob asks something", headers=BOB)
    bob_request = llm.requests[-1]
    assert all("alice" not in m.content for m in bob_request.messages)


async def test_malformed_conversation_id_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/conversations/not-a-uuid", headers=ALICE)
    assert response.status_code == 422
    assert response.json()["error"]["category"] == "invalid_request"

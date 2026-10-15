"""Per-user message limits (spec §18, §22.3 "rate limiting")."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.helpers import (
    ALICE,
    BOB,
    ScriptedLLMClient,
    create_conversation,
    make_settings,
    running_app,
    send,
)


@pytest.fixture
async def limited_client(
    database_url: str, clean_db: None, llm: ScriptedLLMClient
) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_settings(database_url=database_url, user_messages_per_minute=2)
    async with running_app(settings, llm) as (_, client):
        yield client


async def test_messages_beyond_the_per_minute_limit_are_refused(
    limited_client: httpx.AsyncClient,
) -> None:
    cid = await create_conversation(limited_client, ALICE)
    assert (await send(limited_client, cid, "one")).status_code == 201
    assert (await send(limited_client, cid, "two")).status_code == 201

    third = await send(limited_client, cid, "three")
    assert third.status_code == 429
    error = third.json()["error"]
    assert error["category"] == "rate_limited" and error["code"] == "too_many_messages"
    assert 1 <= int(third.headers["retry-after"]) <= 60
    # Nothing was stored for the refused message.
    listed = await limited_client.get(f"/v1/conversations/{cid}/messages", headers=ALICE)
    assert [m["content"] for m in listed.json()["messages"] if m["role"] == "user"] == [
        "one",
        "two",
    ]


async def test_the_limit_counts_across_conversations_and_is_per_user(
    limited_client: httpx.AsyncClient,
) -> None:
    first = await create_conversation(limited_client, ALICE)
    other = await create_conversation(limited_client, ALICE)
    assert (await send(limited_client, first, "a")).status_code == 201
    assert (await send(limited_client, other, "b")).status_code == 201
    assert (await send(limited_client, other, "c")).status_code == 429
    # Bob has his own allowance.
    bobs = await create_conversation(limited_client, BOB)
    assert (await send(limited_client, bobs, "hi", headers=BOB)).status_code == 201


async def test_an_idempotent_retry_is_not_counted_again(
    limited_client: httpx.AsyncClient,
) -> None:
    cid = await create_conversation(limited_client, ALICE)
    assert (await send(limited_client, cid, "one", idempotency_key="k1")).status_code == 201
    assert (await send(limited_client, cid, "two")).status_code == 201
    # The limit is reached, but replaying an existing message still works.
    replay = await send(limited_client, cid, "one", idempotency_key="k1")
    assert replay.status_code in (200, 201)


async def test_a_zero_limit_disables_it(
    database_url: str, clean_db: None, llm: ScriptedLLMClient
) -> None:
    settings = make_settings(
        database_url=database_url, user_messages_per_minute=0, user_messages_per_day=0
    )
    async with running_app(settings, llm) as (_, client):
        cid = await create_conversation(client, ALICE)
        for n in range(4):
            assert (await send(client, cid, f"m{n}")).status_code == 201

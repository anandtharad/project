"""The app in AUTH_MODE=clerk: a Clerk session token creates and reaches the
person's own data; dev tokens and missing or bad tokens are refused. Every
table has row-level security on (migration 0009) and the app still works."""

from __future__ import annotations

import time

import jwt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.helpers import ALICE, ScriptedLLMClient, make_settings, running_app
from tests.unit.test_clerk_auth import KEY, ORIGIN, PUBLIC_PEM


def bearer(sub: str) -> dict[str, str]:
    now = int(time.time())
    claims = {"sub": sub, "exp": now + 60, "nbf": now - 1, "azp": ORIGIN}
    return {"Authorization": f"Bearer {jwt.encode(claims, KEY, algorithm='RS256')}"}


async def test_clerk_tokens_identify_people(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    settings = make_settings(
        database_url=database_url,
        auth_mode="clerk",
        clerk_jwt_key=PUBLIC_PEM,
        clerk_authorized_parties=ORIGIN,
    )
    async with running_app(settings, ScriptedLLMClient()) as (_, client):
        created = await client.post("/v1/conversations", headers=bearer("user_a"))
        assert created.status_code == 201
        cid = created.json()["conversation_id"]
        assert (
            await client.get(f"/v1/conversations/{cid}", headers=bearer("user_a"))
        ).status_code == 200
        # Another Clerk user: the same 404 as a missing conversation.
        assert (
            await client.get(f"/v1/conversations/{cid}", headers=bearer("user_b"))
        ).status_code == 404
        for headers in (ALICE, {}, {"Authorization": "Bearer nope"}):
            assert (await client.post("/v1/conversations", headers=headers)).status_code == 401
    async with db_engine.connect() as conn:
        ids = (await conn.execute(text("SELECT external_id FROM users ORDER BY 1"))).scalars().all()
        unprotected = (
            (
                await conn.execute(
                    text(
                        "SELECT relname FROM pg_class c"
                        " JOIN pg_namespace n ON n.oid = c.relnamespace"
                        " WHERE n.nspname = 'public' AND c.relkind = 'r'"
                        " AND NOT c.relrowsecurity"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert ids == ["clerk:user_a", "clerk:user_b"]
    assert unprotected == []

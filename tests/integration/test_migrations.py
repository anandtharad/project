from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.models import Base
from app.main import migration_head
from tests.conftest import alembic_config, create_temp_database, drop_temp_database


@pytest.fixture
async def scratch_db(base_database_url: str) -> AsyncIterator[str]:
    url = await create_temp_database(base_database_url)
    yield url
    await drop_temp_database(base_database_url, url)


async def test_upgrade_downgrade_upgrade_and_models_match(scratch_db: str) -> None:
    config = alembic_config(scratch_db)
    await asyncio.to_thread(command.upgrade, config, "head")

    engine = create_async_engine(scratch_db)
    try:
        async with engine.connect() as conn:

            def diff(sync_conn: Any) -> list[Any]:
                return list(compare_metadata(MigrationContext.configure(sync_conn), Base.metadata))

            assert await conn.run_sync(diff) == [], "models and migrations have drifted"
            version = await conn.scalar(text("SELECT version_num FROM alembic_version"))
            assert version == migration_head()

        await asyncio.to_thread(command.downgrade, config, "base")
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert set(tables) <= {"alembic_version"}

        await asyncio.to_thread(command.upgrade, config, "head")
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert {"users", "conversations", "messages"} <= set(tables)
    finally:
        await engine.dispose()


async def test_ready_when_migrated(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database": "ok",
            "migrations": "ok",
            "llm_provider": "scripted",
            "prompt": "companion_system@1.1.0",
        },
    }

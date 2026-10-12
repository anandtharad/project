"""Fixtures.

Integration and security tests need PostgreSQL: set TEST_DATABASE_URL to a
database whose role can CREATE DATABASE (e.g. the compose `db` service). Each
test session creates a throwaway database, migrates it with Alembic, and drops
it afterwards. Without TEST_DATABASE_URL those tests are skipped and say so.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from tests.helpers import ScriptedLLMClient, make_settings, running_app, suite_of

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variables that Settings or the provider SDKs read implicitly.
# Tests must not depend on the developer's shell, so these are cleared
# (live tests opt out: they need the real provider key).
_ISOLATED_ENV_PREFIXES = ("GENERATION_", "OPENAI_", "ANTHROPIC_", "GROQ_", "GOOGLE_GENAI_")
_ISOLATED_ENV_NAMES = {
    "MAX_OUTPUT_TOKENS",
    "MAX_CONTEXT_TOKENS",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GEMINI_BASE_URL",
    "XAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "MISTRAL_API_KEY",
}


@pytest.fixture(autouse=True)
def _isolate_environment(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    for name in list(os.environ):
        if name in _ISOLATED_ENV_NAMES or name.startswith(_ISOLATED_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)


def alembic_config(url: str) -> AlembicConfig:
    config = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


async def create_temp_database(base_url: str) -> str:
    url = make_url(base_url)
    name = f"companion_test_{uuid.uuid4().hex[:10]}"
    admin = create_async_engine(url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await admin.dispose()
    return url.set(database=name).render_as_string(hide_password=False)


async def drop_temp_database(base_url: str, temp_url: str) -> None:
    name = make_url(temp_url).database
    admin = create_async_engine(make_url(base_url), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    await admin.dispose()


@pytest.fixture(scope="session")
def base_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; PostgreSQL-backed tests skipped")
    return url


@pytest.fixture(scope="session")
async def database_url(base_database_url: str) -> AsyncIterator[str]:
    url = await create_temp_database(base_database_url)
    await asyncio.to_thread(command.upgrade, alembic_config(url), "head")
    yield url
    await drop_temp_database(base_database_url, url)


@pytest.fixture(scope="session")
async def db_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    yield engine
    await engine.dispose()


@pytest.fixture
async def clean_db(db_engine: AsyncEngine) -> None:
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE users, conversations, messages CASCADE"))


@pytest.fixture
def llm() -> ScriptedLLMClient:
    return ScriptedLLMClient()


@pytest.fixture
async def app_and_client(
    database_url: str, clean_db: None, llm: ScriptedLLMClient
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    async with running_app(make_settings(database_url=database_url), llm) as pair:
        yield pair


@pytest.fixture
def app(app_and_client: tuple[FastAPI, httpx.AsyncClient]) -> FastAPI:
    return app_and_client[0]


@pytest.fixture
def client(app_and_client: tuple[FastAPI, httpx.AsyncClient]) -> httpx.AsyncClient:
    return app_and_client[1]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        suite = suite_of(item.path)
        if suite in ("integration", "security"):
            item.add_marker(pytest.mark.integration)
        if suite == "live":
            item.add_marker(pytest.mark.live)

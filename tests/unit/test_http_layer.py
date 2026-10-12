"""HTTP behavior that needs no database (the app points at an unreachable one)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from app.api.middleware import _innermost_location
from tests.helpers import ALICE, ScriptedLLMClient, make_settings, running_app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with running_app(make_settings(), ScriptedLLMClient()) as (_, client):
        yield client


async def test_health_does_not_need_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_reports_database_unavailable(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"] == "unavailable"


async def test_database_outage_returns_dependency_error(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/conversations", headers=ALICE)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["category"] == "dependency_unavailable"
    assert response.headers["retry-after"]
    assert "127.0.0.1" not in response.text  # no connection details leaked


async def test_trace_id_is_generated_and_echoed(client: httpx.AsyncClient) -> None:
    generated = await client.get("/health")
    assert len(generated.headers["x-request-id"]) == 32
    echoed = await client.get("/health", headers={"X-Request-ID": "client-trace-123"})
    assert echoed.headers["x-request-id"] == "client-trace-123"
    replaced = await client.get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert replaced.headers["x-request-id"] != "bad id <script>"


async def test_unknown_route_uses_error_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["category"] == "not_found"
    assert error["code"] == "route_not_found"
    assert error["trace_id"] == response.headers["x-request-id"]


async def test_missing_credentials_rejected_before_database(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/conversations")
    assert response.status_code == 401
    assert response.json()["error"]["category"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


async def test_declared_oversized_body_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/conversations/00000000-0000-0000-0000-000000000000/messages",
        content=b"x" * 70_000,
        headers={**ALICE, "content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


async def test_streamed_oversized_body_rejected(client: httpx.AsyncClient) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(20):
            yield b"x" * 8192

    response = await client.post(
        "/v1/conversations/00000000-0000-0000-0000-000000000000/messages",
        content=chunks(),
        headers={**ALICE, "content-type": "application/json"},
    )
    assert response.status_code == 413


async def test_openapi_available_outside_production(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/v1/conversations/{conversation_id}/messages" in paths


def test_error_location_is_a_bare_file_name_for_windows_paths() -> None:
    code = compile("raise RuntimeError('boom')", r"C:\anand\project\app\boom.py", "exec")
    try:
        exec(code)  # noqa: S102 - fixed code object with a Windows-style file name
    except RuntimeError as exc:
        assert _innermost_location(exc) == "boom.py:1"

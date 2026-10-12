"""Shared test utilities: settings, a scripted LLM test double, and app runners."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.llm.schemas import (
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    TokenUsage,
)
from app.main import create_app

ALICE = {"Authorization": "Bearer token-alice"}
BOB = {"Authorization": "Bearer token-bob"}
UNREACHABLE_DB = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "log_level": "WARNING",
        "database_url": UNREACHABLE_DB,
        "generation_provider": "fake",
        "dev_auth_tokens": "alice:token-alice,bob:token-bob",
        "turn_deadline_seconds": 10.0,
        "stale_turn_seconds": 30.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_response(
    content: str,
    *,
    finish_reason: FinishReason = FinishReason.STOP,
    model_id: str = "scripted-model",
    requested_model: str = "scripted-model",
) -> GenerationResponse:
    return GenerationResponse(
        content=content,
        provider="scripted",
        requested_model=requested_model,
        model_id=model_id,
        finish_reason=finish_reason,
        raw_finish_reason=finish_reason.value,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        provider_request_id="req_test",
        latency_ms=1.0,
    )


Step = (
    str
    | GenerationResponse
    | LLMError
    | Callable[[GenerationRequest], Awaitable[GenerationResponse | str]]
)


class ScriptedLLMClient:
    """Deterministic LLM test double.

    Each call consumes the next queued step; with an empty queue it replies
    "reply to: <last user message>" so tests can trace which context was used.
    Every request is recorded.
    """

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self._steps: deque[Step] = deque()

    @property
    def provider(self) -> str:
        return "scripted"

    async def aclose(self) -> None:
        return None

    def queue(self, *steps: Step) -> None:
        self._steps.extend(steps)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        step: Step = (
            self._steps.popleft()
            if self._steps
            else f"reply to: {request.messages[-1].content.split(chr(10))[-1]}"
        )
        if isinstance(step, LLMError):
            raise step
        if isinstance(step, GenerationResponse):
            return step
        if isinstance(step, str):
            return make_response(step)
        result = await step(request)
        return make_response(result) if isinstance(result, str) else result


def blocking_step(release: asyncio.Event, entered: asyncio.Event | None = None) -> Step:
    """A step that waits until `release` is set (to hold a turn in flight)."""

    async def step(request: GenerationRequest) -> str:
        if entered is not None:
            entered.set()
        await release.wait()
        return f"reply to: {request.messages[-1].content}"

    return step


@asynccontextmanager
async def running_app(
    settings: Settings, llm: ScriptedLLMClient
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    app = create_app(settings, llm_client=llm)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


async def create_conversation(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post("/v1/conversations", headers=headers)
    assert response.status_code == 201, response.text
    return str(response.json()["conversation_id"])


async def send(
    client: httpx.AsyncClient,
    conversation_id: str,
    content: str,
    *,
    headers: dict[str, str] = ALICE,
    idempotency_key: str | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {"content": content}
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return await client.post(
        f"/v1/conversations/{conversation_id}/messages", json=body, headers=headers
    )


async def list_messages(
    client: httpx.AsyncClient, conversation_id: str, headers: dict[str, str] = ALICE
) -> list[dict[str, Any]]:
    response = await client.get(f"/v1/conversations/{conversation_id}/messages", headers=headers)
    assert response.status_code == 200, response.text
    messages: list[dict[str, Any]] = response.json()["messages"]
    return messages

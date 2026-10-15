"""Shared test utilities: settings, a scripted LLM test double, and app runners."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Any

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.decisions.backend import BackendAnswers, DecisionBackendError
from app.decisions.schemas import DecisionProvider, QuestionSpec
from app.decisions.service import DecisionBackend
from app.llm.schemas import (
    FinishReason,
    GenerationPurpose,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    TokenUsage,
)
from app.main import create_app
from app.prompting import PromptRegistry

ALICE = {"Authorization": "Bearer token-alice"}
BOB = {"Authorization": "Bearer token-bob"}
UNREACHABLE_DB = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"


def suite_of(path: PurePath) -> str | None:
    """The tests/<suite>/ directory a test file is in, on any OS."""
    return path.parent.name if path.parent.parent.name == "tests" else None


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "log_level": "WARNING",
        "database_url": UNREACHABLE_DB,
        "generation_provider": "fake",
        # Deterministic, no model download; tests needing real vectors opt in.
        "embedding_provider": "fake",
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


EMPTY_EXTRACTION = json.dumps(
    {
        "topic": "",
        "subtopics": [],
        "user_goal": {"value": "", "provenance": "unknown"},
        "emotional_context": {"value": "", "provenance": "unknown"},
        "open_questions": [],
        "recently_discussed": [],
    }
)


NO_MEMORIES = json.dumps({"candidates": []})

# Not a valid decision reply: decision tests queue the reply they need.
NO_DECISIONS = "{}"

# The safety classifier's and the output check's default answers.
NO_SAFETY_RISK = json.dumps({"category": "none", "rationale": "Ordinary conversation."})
OUTPUT_PASSES = json.dumps({"verdict": "pass", "rationale": "Meets the pathway."})


class ScriptedLLMClient:
    """Deterministic LLM test double.

    Each conversation call consumes the next queued step; with an empty queue
    it replies "reply to: <last user message>" so tests can trace which
    context was used. Every request is recorded.

    State-extraction, memory-extraction, decision and safety calls are
    recorded separately (`state_requests`, `memory_requests`,
    `decision_requests`, `safety_requests`, `safety_output_requests`) and
    consume their own queues (`queue_state`, `queue_memory`, `queue_decision`,
    `queue_safety`, `queue_safety_output`); by default they return an empty
    state, no memory candidates, an empty (invalid) decision reply, the
    category "none" and the verdict "pass", so scripted conversation replies
    are unaffected.
    """

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.state_requests: list[GenerationRequest] = []
        self.memory_requests: list[GenerationRequest] = []
        self.decision_requests: list[GenerationRequest] = []
        self.safety_requests: list[GenerationRequest] = []
        self.safety_output_requests: list[GenerationRequest] = []
        self.closed = False
        self._steps: deque[Step] = deque()
        self._state_steps: deque[Step] = deque()
        self._memory_steps: deque[Step] = deque()
        self._decision_steps: deque[Step] = deque()
        self._safety_steps: deque[Step] = deque()
        self._safety_output_steps: deque[Step] = deque()

    @property
    def provider(self) -> str:
        return "scripted"

    async def aclose(self) -> None:
        self.closed = True

    def queue(self, *steps: Step) -> None:
        self._steps.extend(steps)

    def queue_state(self, *steps: Step) -> None:
        self._state_steps.extend(steps)

    def queue_memory(self, *steps: Step) -> None:
        self._memory_steps.extend(steps)

    def queue_decision(self, *steps: Step) -> None:
        self._decision_steps.extend(steps)

    def queue_safety(self, *steps: Step) -> None:
        self._safety_steps.extend(steps)

    def queue_safety_output(self, *steps: Step) -> None:
        self._safety_output_steps.extend(steps)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        step: Step
        if request.trace.purpose is GenerationPurpose.STATE_EXTRACTION:
            self.state_requests.append(request)
            step = self._state_steps.popleft() if self._state_steps else EMPTY_EXTRACTION
        elif request.trace.purpose is GenerationPurpose.MEMORY_EXTRACTION:
            self.memory_requests.append(request)
            step = self._memory_steps.popleft() if self._memory_steps else NO_MEMORIES
        elif request.trace.purpose is GenerationPurpose.DECISION:
            self.decision_requests.append(request)
            step = self._decision_steps.popleft() if self._decision_steps else NO_DECISIONS
        elif request.trace.purpose is GenerationPurpose.SAFETY_CLASSIFICATION:
            self.safety_requests.append(request)
            step = self._safety_steps.popleft() if self._safety_steps else NO_SAFETY_RISK
        elif request.trace.purpose is GenerationPurpose.SAFETY_OUTPUT_CHECK:
            self.safety_output_requests.append(request)
            step = (
                self._safety_output_steps.popleft() if self._safety_output_steps else OUTPUT_PASSES
            )
        else:
            self.requests.append(request)
            step = (
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
    settings: Settings, llm: ScriptedLLMClient, decision_backend: DecisionBackend | None = None
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    app = create_app(settings, llm_client=llm, decision_backend=decision_backend)
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


DEFAULT_PROMPT_VERSION: str = Settings.model_fields["companion_prompt_version"].default


def default_system_prompt() -> str:
    """The reply system prompt when no policy constrains the approach."""
    prompt = PromptRegistry().get("companion_system", DEFAULT_PROMPT_VERSION)
    return prompt.render(response_guidance=prompt.fragment("open"))


class ScriptedDecisionBackend:
    """Decision backend test double: returns fixed answers or raises."""

    provider: DecisionProvider = "laya"
    model = "english"

    def __init__(
        self,
        answers: dict[str, dict[str, Any]] | None = None,
        *,
        error: DecisionBackendError | None = None,
    ) -> None:
        self.answers = answers or {}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ask(self, state: Any, questions: list[QuestionSpec]) -> BackendAnswers:
        self.calls.append(dict(state))
        if self.error is not None:
            raise self.error
        return BackendAnswers(
            model_id="laya-test",
            answers=self.answers,
            latency_ms=5.0,
            request_id=None,
            input_tokens=10,
        )

    async def aclose(self) -> None:
        return None


def choice(question: QuestionSpec, label: str, confidence: float = 0.9) -> dict[str, Any]:
    others = (1.0 - 0.9) / (len(question.options) - 1)
    return {
        "type": "choice",
        "choice": label,
        "confidence": confidence,
        "probabilities": {o: (0.9 if o == label else others) for o in question.options},
    }

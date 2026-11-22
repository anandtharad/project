"""The agent graph on its own (app/agent/graph.py): failures raise
AgentFailed for the orchestrator's fallback, the deadline is enforced by
the graph whatever the provider does, and the result carries the messages
a regeneration answers from."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import pytest

from app.agent.graph import AgentContext, AgentDeps, AgentFailed, run_agent
from app.agent.strategy import StrategyChoice, StrategyRegistry
from app.agent.tools import ToolContext
from app.agent.tracing import warn_if_tracing
from app.config import AppEnv
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    ToolCall,
    TraceMetadata,
)
from app.prompting import PromptRegistry
from tests.helpers import ScriptedLLMClient, make_response, make_settings, tool_call_response

REGISTRY = StrategyRegistry(PromptRegistry().get("agent_strategies", "1.0.0"))
HISTORY = [ChatMessage(role=ChatRole.USER, content="hello")]


def deps(llm: ScriptedLLMClient, *, seconds: float = 10.0, **limits: Any) -> AgentDeps:
    def build(choice: StrategyChoice, final: bool) -> AgentContext:
        system = f"SYSTEM {choice.strategy.id}" + (" FINAL" if final else "")
        return AgentContext(system, list(HISTORY))

    return AgentDeps(
        llm=llm,
        model="m",
        params=GenerationParams(),
        max_output_tokens=256,
        trace=TraceMetadata(trace_id="t", purpose=GenerationPurpose.AGENT_STEP),
        registry=REGISTRY,
        tools=ToolContext(
            user_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            context_from_sequence=None,
            state=None,
            memory=None,
            knowledge=None,
            session_factory=None,  # type: ignore[arg-type]
        ),
        history=HISTORY,
        notes=None,
        build=build,
        deadline=time.monotonic() + seconds,
        **limits,
    )


async def test_the_result_carries_the_final_messages_and_system() -> None:
    llm = ScriptedLLMClient()
    llm.queue(
        tool_call_response(ToolCall(id="a", name="get_conversation_state", arguments="{}")),
        "hi!",
    )
    result = await run_agent(deps(llm))
    assert result.response.content == "hi!" and result.system == "SYSTEM conversational"
    assert [m.role for m in result.messages] == [ChatRole.USER, ChatRole.ASSISTANT, ChatRole.TOOL]
    assert result.record["status"] == "completed" and result.record["model_calls"] == 2


async def test_after_the_limit_the_result_has_the_final_answer_system() -> None:
    llm = ScriptedLLMClient()
    llm.queue(
        tool_call_response(ToolCall(id="a", name="get_conversation_state", arguments="{}")),
        "done",
    )
    result = await run_agent(deps(llm, max_iterations=1))
    assert result.system == "SYSTEM conversational FINAL"
    assert result.record["status"] == "completed_after_limit"


async def test_a_tool_calls_finish_without_calls_is_malformed() -> None:
    llm = ScriptedLLMClient()
    llm.queue(make_response("", finish_reason=FinishReason.TOOL_CALLS))
    with pytest.raises(AgentFailed) as info:
        await run_agent(deps(llm))
    assert info.value.reason == "malformed_tool_calls"


async def test_the_graph_enforces_the_deadline_on_a_slow_provider() -> None:
    llm = ScriptedLLMClient()

    async def slow(request: GenerationRequest) -> str:
        await asyncio.sleep(5)  # a provider that ignores timeout_seconds
        return "late"

    llm.queue(slow)
    started = time.monotonic()
    with pytest.raises(AgentFailed) as info:
        await run_agent(deps(llm, seconds=0.3))
    assert info.value.reason == "deadline" and time.monotonic() - started < 2


async def test_an_expired_deadline_fails_before_any_call() -> None:
    llm = ScriptedLLMClient()
    with pytest.raises(AgentFailed) as info:
        await run_agent(deps(llm, seconds=-1))
    assert info.value.reason == "deadline"
    assert llm.strategy_requests == [] and llm.requests == []


def test_langsmith_tracing_is_refused_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    warn_if_tracing(make_settings())  # only a warning outside production
    with pytest.raises(ValueError, match="LangSmith"):
        warn_if_tracing(make_settings().model_copy(update={"app_env": AppEnv.PRODUCTION}))

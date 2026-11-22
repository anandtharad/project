"""The agent turn as a LangGraph StateGraph (docs/agent.md, "Graph").

    prepare -> agent -> (tools -> agent)* -> finalize

- prepare: the strategy (lean: already named by the state-extraction call;
  full: a separate structured call, app/agent/strategy.py); the
  orchestrator's builder turns it into the trusted system prompt and the
  context (ContextBuilder), with no memories or evidence up front.
- agent: one model call with the tool definitions. A reply with tool calls
  goes to `tools`; a reply without goes to `finalize`.
- tools: runs the requested calls (app/agent/tools.py) within the turn's
  budget and appends the assistant message and one result per call.
- finalize: the last reply, its strategy, the collected evidence and the
  execution record go back to the orchestrator, which validates, checks
  safety and citations, and persists, exactly as for any other reply.

Limits: at most `max_iterations` model calls that may use tools and
`max_tool_calls` tool calls per turn; then one last call with tool_choice
"none" and the final-answer note. Every model call and tool call is cut by
the turn's remaining deadline. The graph has no checkpointer: PostgreSQL
stays the only record of the conversation. Services live in the run's
context (`AgentDeps`), never in the graph state.

Failures: a provider error, a deadline, or a model that still wants tools
after the limit raises `AgentFailed`; the orchestrator then generates the
reply the deterministic way. The agent never decides safety: safety
pathways do not reach it, and its reply goes through the same checks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agent.strategy import StrategyChoice, StrategyRegistry, select_strategy
from app.agent.tools import ToolContext, ToolResult, execute, tool_specs
from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    ToolCall,
    TraceMetadata,
)


class AgentFailed(Exception):
    """The agent could not produce a reply; `reason` is safe to log."""

    def __init__(self, reason: str, record: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record = record


@dataclass(frozen=True)
class AgentContext:
    """What the orchestrator builds for the chosen strategy."""

    system: str
    messages: list[ChatMessage]


@dataclass
class AgentDeps:
    llm: LLMClient
    model: str
    params: GenerationParams
    max_output_tokens: int
    trace: TraceMetadata
    registry: StrategyRegistry
    tools: ToolContext
    # The recent history and notes the strategy choice reads.
    history: list[ChatMessage]
    notes: dict[str, Any] | None
    # Builds the trusted system prompt and the context for a strategy.
    build: Callable[[StrategyChoice, bool], AgentContext]
    deadline: float  # time.monotonic() value
    # The lean agent's strategy, already chosen by the state-extraction
    # call; None: prepare makes the separate strategy call (full profile).
    strategy: StrategyChoice | None = None
    # The tools offered (None: all four, the full profile).
    tool_names: tuple[str, ...] | None = None
    max_iterations: int = 4
    max_tool_calls: int = 6
    tool_timeout_seconds: float = 5.0
    tool_max_characters: int = 4000
    steps: list[dict[str, Any]] = field(default_factory=list)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


class AgentState(TypedDict, total=False):
    strategy: StrategyChoice
    system: str
    messages: list[ChatMessage]
    iterations: int
    tool_calls: int
    pending: list[ToolCall]
    final: GenerationResponse
    status: str


def _record(deps: AgentDeps, state: AgentState, status: str) -> dict[str, Any]:
    tools = [s for s in deps.steps if s["kind"] == "tool"]
    calls = [s for s in deps.steps if s["kind"] == "model"]

    def total(key: str) -> int | None:
        values = [s[key] for s in calls if s.get(key) is not None]
        return sum(values) if values else None

    strategy = state.get("strategy")
    return {
        "status": status,
        "strategy": strategy.summary() if strategy is not None else None,
        "iterations": state.get("iterations", 0),
        "tool_calls": state.get("tool_calls", 0),
        "tools_called": [t["tool"] for t in tools],
        "tools": [{k: v for k, v in t.items() if k != "kind"} for t in tools],
        "model_calls": len(calls),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "evidence_items": len(deps.tools.evidence.items),
    }


async def prepare(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    if deps.strategy is not None:
        context = deps.build(deps.strategy, False)
        return {
            "strategy": deps.strategy,
            "system": context.system,
            "messages": list(context.messages),
            "iterations": 0,
            "tool_calls": 0,
            "pending": [],
        }
    remaining = deps.remaining()
    if remaining <= 0:
        raise AgentFailed("deadline", _record(deps, state, "deadline"))
    try:
        # The provider timeout and the turn's deadline, whichever comes first.
        async with asyncio.timeout(remaining):
            choice = await select_strategy(
                deps.registry,
                deps.llm,
                model=deps.model,
                history=deps.history,
                notes=deps.notes,
                timeout_seconds=max(1.0, min(30.0, remaining)),
                params=deps.params,
                trace=deps.trace,
            )
    except TimeoutError as exc:
        raise AgentFailed("deadline", _record(deps, state, "deadline")) from exc
    context = deps.build(choice, False)
    return {
        "strategy": choice,
        "system": context.system,
        "messages": list(context.messages),
        "iterations": 0,
        "tool_calls": 0,
        "pending": [],
    }


async def agent(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    iterations = state.get("iterations", 0)
    exhausted = (
        iterations >= deps.max_iterations or state.get("tool_calls", 0) >= deps.max_tool_calls
    )
    system = state["system"]
    if exhausted:
        system = deps.build(state["strategy"], True).system
    remaining = deps.remaining()
    if remaining <= 0:
        raise AgentFailed("deadline", _record(deps, state, "deadline"))
    request = GenerationRequest(
        model=deps.model,
        system=system,
        messages=state["messages"],
        max_output_tokens=deps.max_output_tokens,
        timeout_seconds=remaining,
        params=deps.params,
        trace=deps.trace.model_copy(update={"purpose": GenerationPurpose.AGENT_STEP}),
        tools=tool_specs(deps.tool_names),
        tool_choice="none" if exhausted else "auto",
    )
    try:
        async with asyncio.timeout(remaining):
            response = await deps.llm.generate(request)
    except LLMError as exc:
        raise AgentFailed(
            f"provider:{exc.category.value}", _record(deps, state, "provider_error")
        ) from exc
    except TimeoutError as exc:
        raise AgentFailed("deadline", _record(deps, state, "deadline")) from exc
    deps.steps.append(
        {
            "kind": "model",
            "iteration": iterations + 1,
            "finish_reason": response.finish_reason.value,
            "tool_calls": [c.name for c in response.tool_calls],
            "latency_ms": round(response.latency_ms, 1),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    )
    if response.tool_calls:
        if exhausted:
            raise AgentFailed("tools_after_limit", _record(deps, state, "iteration_limit"))
        assistant = ChatMessage(
            role=ChatRole.ASSISTANT, content=response.content, tool_calls=response.tool_calls
        )
        return {
            "messages": [*state["messages"], assistant],
            "pending": list(response.tool_calls),
            "iterations": iterations + 1,
        }
    if response.finish_reason is FinishReason.TOOL_CALLS:
        raise AgentFailed("malformed_tool_calls", _record(deps, state, "malformed"))
    return {
        "final": response,
        "pending": [],
        "iterations": iterations + 1,
        "status": "completed_after_limit" if exhausted else "completed",
    }


async def tools(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    used = state.get("tool_calls", 0)
    messages = list(state["messages"])
    for call in state.get("pending", []):
        if used >= deps.max_tool_calls:
            result = ToolResult(
                call.name,
                "unavailable",
                '{"status": "error", "error": "limit_reached", '
                '"message": "The app allows no more lookups for this reply."}',
                0.0,
            )
        else:
            used += 1
            timeout = min(deps.tool_timeout_seconds, max(0.1, deps.remaining()))
            result = await execute(
                call, deps.tools, timeout_seconds=timeout, max_characters=deps.tool_max_characters
            )
        deps.steps.append({"kind": "tool", **result.summary()})
        messages.append(
            ChatMessage(
                role=ChatRole.TOOL, content=result.content, tool_call_id=call.id, name=call.name
            )
        )
    return {"messages": messages, "pending": [], "tool_calls": used}


def route(state: AgentState) -> str:
    return "tools" if state.get("pending") else "finalize"


async def finalize(state: AgentState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    return {"status": state.get("status", "completed")}


def build_graph() -> Any:
    graph = StateGraph(AgentState, context_schema=AgentDeps)
    graph.add_node("prepare", prepare)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


GRAPH = build_graph()


@dataclass(frozen=True)
class AgentResult:
    response: GenerationResponse
    strategy: StrategyChoice
    system: str
    # The final call's messages (history, tool calls and results), for a
    # regeneration that answers from the same material.
    messages: list[ChatMessage]
    record: dict[str, Any]


async def run_agent(deps: AgentDeps) -> AgentResult:
    started = time.perf_counter()
    state: AgentState = await GRAPH.ainvoke(
        {},
        config={"recursion_limit": 4 * deps.max_iterations + 10},
        context=deps,
    )
    final = state.get("final")
    if final is None:
        raise AgentFailed("no_answer", _record(deps, state, "no_answer"))
    record = _record(deps, state, state.get("status", "completed"))
    record["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    # The system prompt of the call that produced the reply (with the
    # final-answer note when the limit was reached).
    system = (
        runtime_system(deps, state)
        if state.get("status") == "completed_after_limit"
        else state["system"]
    )
    return AgentResult(final, state["strategy"], system, list(state["messages"]), record)


def runtime_system(deps: AgentDeps, state: AgentState) -> str:
    return deps.build(state["strategy"], True).system

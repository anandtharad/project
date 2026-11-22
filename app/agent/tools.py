"""Read-only tools for agent turns (docs/agent.md, "Tools").

The model sees each tool's name, description and argument schema, and
chooses whether to call it. Everything else is application code:

- **Identity is not an argument.** The person and conversation come from the
  turn's `ToolContext`, built after authentication and the ownership check;
  argument schemas forbid unknown fields, so a `user_id` the model invents is
  rejected as invalid arguments, never used.
- **Existing services, existing policies.** Memories come from
  `MemoryService.retrieve_for_turn` (memory settings, statuses, retention),
  knowledge from `KnowledgeService.search` (hybrid retrieval and relevance
  thresholds), notes from the turn's conversation state, history from an
  owner-scoped full-text query.
- **Untrusted output.** Results use the blocks the companion prompt already
  treats as data, never instructions: `<memories>`, `<evidence>`,
  `<conversation_notes>`, and `<conversation_history>`; `<` and `>` inside
  them are escaped so stored text cannot close a block.
- **Bounded.** Each call has a time limit and a size limit; failures come
  back to the model as a normalized error, never as invented results.
- **Logged without content:** tool, status, duration, result count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.context.builder import render_memories, render_notes
from app.db import repositories as repo
from app.llm.schemas import ToolCall, ToolSpec
from app.memory.schemas import MemoryStatus
from app.memory.service import MemoryService
from app.observability import log_event
from app.rag.evidence import EvidenceItem, render_evidence
from app.rag.retrieval import Passage
from app.rag.service import KnowledgeService
from app.state.schemas import ConversationState

logger = logging.getLogger("app.agent")

ToolStatus = Literal[
    "ok", "empty", "disabled", "invalid_arguments", "unknown_tool", "timeout", "unavailable"
]
HISTORY_RESULTS = 5
HISTORY_SNIPPET_CHARACTERS = 300
MAX_EVIDENCE_ITEMS = 8


def _escaped_json(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, indent=1)
    return body.replace("<", "\\u003c").replace(">", "\\u003e")


@dataclass
class EvidenceCollector:
    """Knowledge passages retrieved during one agent turn, numbered S1, S2...
    across every search, so each marker is unique in the turn and the
    citation check after the reply (app/rag/evidence.py) sees them all."""

    items: list[EvidenceItem] = field(default_factory=list)

    def add(self, passages: list[Passage]) -> list[EvidenceItem]:
        out = []
        known = {item.passage.candidate.chunk_id: item for item in self.items}
        for passage in passages:
            existing = known.get(passage.candidate.chunk_id)
            if existing is not None:
                out.append(existing)
                continue
            if len(self.items) >= MAX_EVIDENCE_ITEMS:
                break
            item = EvidenceItem(f"S{len(self.items) + 1}", passage)
            self.items.append(item)
            known[passage.candidate.chunk_id] = item
            out.append(item)
        return out


@dataclass
class ToolContext:
    """What tools may use in one turn. Built by the orchestrator from the
    authenticated request; never shown to the model."""

    user_id: uuid.UUID
    conversation_id: uuid.UUID
    # The first message sequence already in the model's context: the history
    # tool leaves the current conversation's messages from here on out.
    context_from_sequence: int | None
    state: ConversationState | None
    memory: MemoryService | None
    knowledge: KnowledgeService | None
    session_factory: async_sessionmaker[AsyncSession]
    evidence: EvidenceCollector = field(default_factory=EvidenceCollector)


class QueryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=300)


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class ToolResult:
    name: str
    status: ToolStatus
    content: str  # what the model receives
    duration_ms: float
    count: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "count": self.count,
        }


Outcome = tuple[ToolStatus, str, int]  # status, content for the model, result count
Runner = Callable[[Any, ToolContext], Awaitable[Outcome]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Runner
    parameters: dict[str, Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


def _error(code: ToolStatus, message: str) -> str:
    return _escaped_json({"status": "error", "error": code, "message": message})


async def _memories(args: QueryArgs, ctx: ToolContext) -> Outcome:
    if ctx.memory is None:
        return "disabled", _error("disabled", "Memory is turned off."), 0
    result = await ctx.memory.retrieve_for_turn(ctx.user_id, args.query)
    if result.mode == "disabled":
        return "disabled", _error("disabled", "The person has turned memory off."), 0
    if result.mode == "unavailable":
        return "unavailable", _error("unavailable", "Memories are unavailable right now."), 0
    if not result.memories:
        return "empty", render_memories([]), 0
    shown = [
        {
            "type": r.memory.memory_type.value,
            "content": r.memory.content,
            "noted": r.memory.created_at.date().isoformat(),
            **(
                {"status": "unconfirmed"}
                if r.memory.status is MemoryStatus.PENDING_CONFIRMATION
                else {}
            ),
        }
        for r in result.memories
    ]
    return "ok", render_memories(shown), len(shown)


async def _knowledge(args: QueryArgs, ctx: ToolContext) -> Outcome:
    if ctx.knowledge is None:
        return "disabled", _error("disabled", "The knowledge library is turned off."), 0
    outcome = await ctx.knowledge.search(args.query)
    if outcome is None or outcome.status == "unavailable":
        return "unavailable", render_evidence((), "unavailable"), 0
    items = ctx.evidence.add(list(outcome.passages))
    if not items:
        return "empty", render_evidence((), "no_relevant_sources"), 0
    return "ok", render_evidence(items, "ok"), len(items)


async def _history(args: QueryArgs, ctx: ToolContext) -> Outcome:
    try:
        async with ctx.session_factory() as session:
            hits = await repo.search_user_messages(
                session,
                ctx.user_id,
                args.query,
                exclude_conversation_id=ctx.conversation_id,
                exclude_from_sequence=ctx.context_from_sequence,
                limit=HISTORY_RESULTS,
            )
    except (SQLAlchemyError, OSError):
        return "unavailable", _error("unavailable", "History search is unavailable."), 0
    found = [
        {
            "when": hit.created_at.date().isoformat(),
            "this_conversation": hit.conversation_id == ctx.conversation_id,
            "speaker": "person" if hit.role == "user" else "companion",
            "text": hit.content[:HISTORY_SNIPPET_CHARACTERS],
        }
        for hit in hits
    ]
    body = _escaped_json({"status": "ok" if found else "no_matches", "messages": found})
    content = f"<conversation_history>\n{body}\n</conversation_history>"
    return ("ok" if found else "empty"), content, len(found)


async def _state(args: NoArgs, ctx: ToolContext) -> Outcome:
    if ctx.state is None or ctx.state.is_empty():
        return "empty", render_notes({}), 0
    notes = ctx.state.notes()
    return "ok", render_notes(notes), len(notes)


def _query_schema(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"query": {"type": "string", "description": description}},
        "required": ["query"],
        "additionalProperties": False,
    }


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "search_user_memories",
            "Search what this person told you in earlier conversations (their own "
            "long-term memories: facts, preferences, events). Use it when their "
            "message refers to something from before or knowing them would change "
            "your reply.",
            QueryArgs,
            _memories,
            _query_schema("What to look for, in a few words."),
        ),
        Tool(
            "search_knowledge_base",
            "Search the app's small reviewed library of wellbeing information. Use "
            "it when they ask for information or facts; cite passages by their "
            "marker, like [S1]. Not for ordinary conversation.",
            QueryArgs,
            _knowledge,
            _query_schema("The information being asked for."),
        ),
        Tool(
            "search_conversation_history",
            "Search this person's earlier messages, in this and their other "
            "conversations, for something specific they said that is not in the "
            "messages you can already see.",
            QueryArgs,
            _history,
            _query_schema("Words that would appear in what was said."),
        ),
        Tool(
            "get_conversation_state",
            "Get the app's running notes about this conversation (topic, goal, "
            "emotional context). Inferred by the app, possibly wrong.",
            NoArgs,
            _state,
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        ),
    )
}


def tool_specs() -> tuple[ToolSpec, ...]:
    return tuple(tool.spec() for tool in TOOLS.values())


def _bounded(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    # Cut inside the block, then say so; the model never gets an open block.
    head, _, _ = content[: max(0, limit - 80)].rpartition("\n")
    closing = content.rsplit("\n", 1)[-1] if content.startswith("<") else ""
    return f"{head}\n(truncated: the result was longer than the app allows)\n{closing}"


async def execute(
    call: ToolCall, ctx: ToolContext, *, timeout_seconds: float, max_characters: int
) -> ToolResult:
    """Run one tool call. Never raises: every failure is a ToolResult the
    model can read and the turn can record."""
    started = time.perf_counter()

    def done(status: ToolStatus, content: str, count: int = 0) -> ToolResult:
        result = ToolResult(
            call.name,
            status,
            _bounded(content, max_characters),
            (time.perf_counter() - started) * 1000,
            count,
        )
        log_event(logger, "agent.tool", **result.summary())
        return result

    tool = TOOLS.get(call.name)
    if tool is None:
        return done("unknown_tool", _error("unknown_tool", "There is no tool with that name."))
    try:
        args = tool.args.model_validate_json(call.arguments or "{}")
    except ValidationError:
        return done(
            "invalid_arguments", _error("invalid_arguments", "The arguments are not valid.")
        )
    try:
        async with asyncio.timeout(timeout_seconds):
            status, content, count = await tool.run(args, ctx)
    except TimeoutError:
        return done("timeout", _error("timeout", "The tool took too long."))
    except Exception as exc:  # a tool must never fail the turn
        log_event(logger, "agent.tool_failed", level=logging.WARNING, exc_type=type(exc).__name__)
        return done("unavailable", _error("unavailable", "The tool failed."))
    return done(status, content, count)

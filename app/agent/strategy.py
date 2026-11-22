"""Response strategies for agent turns (docs/agent.md, "Strategies").

The strategies, their descriptions and their guidance are fragments of the
versioned, checksum-locked prompt `agent_strategies` (prompts/); nothing
here holds prompt text. The model picks one per turn in a small structured
call whose schema allows only the registry's ids. The application validates
the answer anyway: an id outside the registry, a reply that does not parse,
or a provider error falls back to `conversational`, recorded as a fallback.
The model never writes instructions: it only names a strategy, and the
strategy's text is the registry's.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    ResponseSchema,
    TraceMetadata,
)
from app.observability import log_event
from app.prompting import PromptTemplate

logger = logging.getLogger("app.agent")

STRATEGY_IDS = ("empathetic", "supportive", "reflective", "informational", "conversational")
DEFAULT_STRATEGY = "conversational"
# Recent messages shown to the selection call (most recent last).
SELECTION_HISTORY = 4
SELECTION_MESSAGE_CHARACTERS = 600
SELECTION_MAX_OUTPUT_TOKENS = 512


@dataclass(frozen=True)
class Strategy:
    id: str
    version: str  # the agent_strategies prompt version
    description: str
    guidance: str  # prompt content, with its response constraints


class StrategyRegistry:
    """The strategies of one `agent_strategies` prompt version."""

    def __init__(self, prompt: PromptTemplate) -> None:
        missing = [
            name
            for sid in STRATEGY_IDS
            for name in (sid, f"{sid}__about")
            if name not in prompt.fragments
        ]
        missing += [n for n in ("agent_tools", "final_answer") if n not in prompt.fragments]
        if missing:
            raise ValueError(f"{prompt.key} lacks fragments: {', '.join(missing)}")
        self.prompt = prompt
        self.strategies = {
            sid: Strategy(
                sid, prompt.version, prompt.fragment(f"{sid}__about"), prompt.fragment(sid)
            )
            for sid in STRATEGY_IDS
        }

    def get(self, strategy_id: str) -> Strategy | None:
        return self.strategies.get(strategy_id)

    def selection_system(self) -> str:
        lines = "\n".join(f"- {s.id}: {s.description}" for s in self.strategies.values())
        return self.prompt.render(strategies=lines)

    def schema(self) -> ResponseSchema:
        return ResponseSchema(
            name="strategy_choice",
            json_schema={
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": list(self.strategies)},
                    "reason": {"type": "string"},
                },
                "required": ["strategy", "reason"],
                "additionalProperties": False,
            },
        )

    @property
    def tool_guidance(self) -> str:
        return self.prompt.fragment("agent_tools")

    @property
    def final_answer(self) -> str:
        return self.prompt.fragment("final_answer")


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str
    reason: str = Field(default="", max_length=500)


@dataclass(frozen=True)
class StrategyChoice:
    strategy: Strategy
    source: Literal["model", "fallback"]
    failure: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def summary(self) -> dict[str, Any]:
        # The model's reason is not stored: it can quote the person.
        return {
            "strategy": self.strategy.id,
            "version": self.strategy.version,
            "source": self.source,
            "failure": self.failure,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _escape(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def selection_input(history: Sequence[ChatMessage], notes: dict[str, Any] | None) -> str:
    """The untrusted data the selection call reads: recent messages and the
    app's notes, escaped and labelled."""
    recent = [m for m in history if m.role in (ChatRole.USER, ChatRole.ASSISTANT)]
    shown = [
        {
            "from": "person" if m.role is ChatRole.USER else "companion",
            "text": m.content[:SELECTION_MESSAGE_CHARACTERS],
        }
        for m in recent[-SELECTION_HISTORY:]
    ]
    body = {"recent_messages": shown, "conversation_notes": notes or {}}
    data = _escape(json.dumps(body, ensure_ascii=False, indent=1))
    return (
        "<conversation_data>\n" + data + "\n</conversation_data>\n"
        "Choose the strategy for the companion's reply to the last message from the person."
    )


async def select_strategy(
    registry: StrategyRegistry,
    llm: LLMClient,
    *,
    model: str,
    history: Sequence[ChatMessage],
    notes: dict[str, Any] | None,
    timeout_seconds: float,
    params: GenerationParams,
    trace: TraceMetadata,
) -> StrategyChoice:
    fallback = registry.strategies[DEFAULT_STRATEGY]
    request = GenerationRequest(
        model=model,
        system=registry.selection_system(),
        messages=[ChatMessage(role=ChatRole.USER, content=selection_input(history, notes))],
        max_output_tokens=SELECTION_MAX_OUTPUT_TOKENS,
        timeout_seconds=timeout_seconds,
        params=params,
        trace=trace.model_copy(update={"purpose": GenerationPurpose.STRATEGY_SELECTION}),
        response_schema=registry.schema(),
    )
    try:
        response = await llm.generate(request)
    except LLMError as exc:
        log_event(logger, "agent.strategy_failed", level=logging.WARNING, error=exc.category.value)
        return StrategyChoice(fallback, "fallback", f"provider:{exc.category.value}")

    def made(
        strategy: Strategy, source: Literal["model", "fallback"], failure: str | None
    ) -> StrategyChoice:
        return StrategyChoice(
            strategy,
            source,
            failure,
            latency_ms=response.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    try:
        choice = _Choice.model_validate_json(response.content)
    except ValidationError:
        log_event(logger, "agent.strategy_invalid", level=logging.WARNING, reason="malformed")
        return made(fallback, "fallback", "malformed")
    chosen = registry.get(choice.strategy)
    if chosen is None:
        # Never executed: only registry strategies have guidance.
        log_event(logger, "agent.strategy_invalid", level=logging.WARNING, reason="unknown_id")
        return made(fallback, "fallback", "unknown_strategy")
    return made(chosen, "model", None)

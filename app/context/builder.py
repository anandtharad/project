"""Assemble the model context for a turn.

Context = privileged system prompt (instructions + the code-chosen response
policy) + recent conversation messages. Untrusted content - user messages and
the conversation notes derived from them (spec §13.2) - only ever travels in
user-role messages; it is never spliced into the system instruction. The
notes open the first included user message inside a <conversation_notes>
block whose JSON escapes "<" and ">", so nothing in it can close the block.
Retrieved memories (also untrusted) open the current user message inside a
<memories> block, escaped the same way, following the spec's context order
(§13.1: state, recent messages, memories, current message).

Token counts are estimated (≈4 characters per token plus per-message
overhead). This is a budgeting heuristic, not an exact provider count.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.llm.schemas import ChatMessage, ChatRole

_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _escaped_json(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, indent=1)
    return body.replace("<", "\\u003c").replace(">", "\\u003e")


def render_notes(notes: Mapping[str, Any]) -> str:
    return f"<conversation_notes>\n{_escaped_json(dict(notes))}\n</conversation_notes>"


def render_memories(memories: Sequence[Mapping[str, Any]]) -> str:
    return f"<memories>\n{_escaped_json([dict(m) for m in memories])}\n</memories>"


@dataclass(frozen=True)
class BuiltContext:
    system: str
    messages: list[ChatMessage]
    included_messages: int
    omitted_messages: int
    estimated_tokens: int
    notes_included: bool = False
    memories_included: int = 0


class ContextBuilder:
    def __init__(self, *, max_context_tokens: int, max_history_messages: int) -> None:
        self._max_context_tokens = max_context_tokens
        self._max_history_messages = max_history_messages

    def build(
        self,
        *,
        system: str,
        history: Sequence[ChatMessage],
        notes: Mapping[str, Any] | None = None,
        memories: Sequence[Mapping[str, Any]] | None = None,
    ) -> BuiltContext:
        """`history` is ordered oldest-first and ends with the current user message."""
        if not history or history[-1].role is not ChatRole.USER:
            raise ValueError("history must end with the current user message")

        notes_block = render_notes(notes) if notes else None
        used = estimate_tokens(system)
        if notes_block is not None:
            used += estimate_tokens(notes_block)
        memories_block = render_memories(memories) if memories else None
        if memories_block is not None:
            used += estimate_tokens(memories_block)
        selected: list[ChatMessage] = []
        # Walk backwards from the current message; it is always included.
        for message in reversed(history):
            cost = estimate_tokens(message.content) + _MESSAGE_OVERHEAD_TOKENS
            if selected and (
                len(selected) >= self._max_history_messages
                or used + cost > self._max_context_tokens
            ):
                break
            selected.append(message)
            used += cost
        selected.reverse()

        # The conversation sent to the model must start with a user message.
        while selected and selected[0].role is not ChatRole.USER:
            used -= estimate_tokens(selected[0].content) + _MESSAGE_OVERHEAD_TOKENS
            selected.pop(0)

        merged = _merge_consecutive(selected)
        if memories_block is not None:
            last = merged[-1]
            merged[-1] = ChatMessage(role=last.role, content=f"{memories_block}\n\n{last.content}")
        if notes_block is not None:
            first = merged[0]
            merged[0] = ChatMessage(role=first.role, content=f"{notes_block}\n\n{first.content}")
        return BuiltContext(
            system=system,
            messages=merged,
            included_messages=len(selected),
            omitted_messages=len(history) - len(selected),
            estimated_tokens=used,
            notes_included=notes_block is not None,
            memories_included=len(memories) if memories else 0,
        )


def _merge_consecutive(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Join adjacent same-role messages (e.g. a user message whose turn failed,
    followed by the next user message) so roles strictly alternate."""
    merged: list[ChatMessage] = []
    for message in messages:
        if merged and merged[-1].role is message.role:
            merged[-1] = ChatMessage(
                role=message.role, content=f"{merged[-1].content}\n\n{message.content}"
            )
        else:
            merged.append(message)
    return merged

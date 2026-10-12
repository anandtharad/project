"""Assemble the model context for a turn.

Phase 1 context = privileged system prompt + recent conversation messages.
User content only ever travels in user-role messages; it is never spliced
into the system instruction.

Token counts are estimated (≈4 characters per token plus per-message
overhead). This is a budgeting heuristic, not an exact provider count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.schemas import ChatMessage, ChatRole

_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class BuiltContext:
    system: str
    messages: list[ChatMessage]
    included_messages: int
    omitted_messages: int
    estimated_tokens: int


class ContextBuilder:
    def __init__(self, *, max_context_tokens: int, max_history_messages: int) -> None:
        self._max_context_tokens = max_context_tokens
        self._max_history_messages = max_history_messages

    def build(self, *, system: str, history: Sequence[ChatMessage]) -> BuiltContext:
        """`history` is ordered oldest-first and ends with the current user message."""
        if not history or history[-1].role is not ChatRole.USER:
            raise ValueError("history must end with the current user message")

        used = estimate_tokens(system)
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
        return BuiltContext(
            system=system,
            messages=merged,
            included_messages=len(selected),
            omitted_messages=len(history) - len(selected),
            estimated_tokens=used,
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

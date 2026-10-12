from __future__ import annotations

import pytest

from app.context.builder import ContextBuilder, estimate_tokens
from app.llm.schemas import ChatMessage, ChatRole


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.USER, content=text)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.ASSISTANT, content=text)


def builder(max_tokens: int = 10_000, max_messages: int = 50) -> ContextBuilder:
    return ContextBuilder(max_context_tokens=max_tokens, max_history_messages=max_messages)


def test_small_history_is_kept_in_order() -> None:
    history = [user("u1"), assistant("a1"), user("u2")]
    built = builder().build(system="sys", history=history)
    assert built.system == "sys"
    assert built.messages == history
    assert built.omitted_messages == 0


def test_message_cap_keeps_most_recent() -> None:
    history = [user("u1"), assistant("a1"), user("u2"), assistant("a2"), user("u3")]
    built = builder(max_messages=3).build(system="sys", history=history)
    assert [m.content for m in built.messages] == ["u2", "a2", "u3"]
    assert built.omitted_messages == 2


def test_token_budget_drops_oldest_and_keeps_user_first() -> None:
    history = [user("x" * 400), assistant("y" * 400), user("short")]
    # Room for the system prompt + current message + the assistant turn only;
    # the leading assistant message is then dropped so the context starts with a user turn.
    budget = estimate_tokens("sys") + estimate_tokens("short") + estimate_tokens("y" * 400) + 8
    built = builder(max_tokens=budget).build(system="sys", history=history)
    assert [m.content for m in built.messages] == ["short"]
    assert built.messages[0].role is ChatRole.USER


def test_current_message_always_included() -> None:
    built = builder(max_tokens=1).build(system="sys", history=[user("hello" * 100)])
    assert len(built.messages) == 1


def test_consecutive_user_messages_are_merged() -> None:
    history = [user("first (turn failed)"), user("second")]
    built = builder().build(system="sys", history=history)
    assert built.messages == [user("first (turn failed)\n\nsecond")]
    assert built.included_messages == 2


def test_history_must_end_with_user_message() -> None:
    with pytest.raises(ValueError):
        builder().build(system="sys", history=[user("u"), assistant("a")])
    with pytest.raises(ValueError):
        builder().build(system="sys", history=[])

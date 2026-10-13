from __future__ import annotations

import pytest

from app.llm.reasoning import strip_reasoning


@pytest.mark.parametrize(
    ("raw", "expected", "removed"),
    [
        ("No reasoning here.", "No reasoning here.", False),
        ("Use < and > freely", "Use < and > freely", False),
        ("<think>plan</think>\n\nHello there.", "Hello there.", True),
        ("<THINK>x\ny</THINK>Hi", "Hi", True),
        # Opening tag injected by the chat template: only the closer is emitted.
        ("The user is sad. Be warm.\n</think>\n\nThat sounds hard.", "That sounds hard.", True),
        ("reasoning</THINK>Answer", "Answer", True),
        # Non-greedy: text between two blocks survives.
        ("<think>a</think>keep<think>b</think> this", "keep this", True),
        # Output cut off mid-reasoning: nothing may be shown.
        ("<think>cut off mid reasoning", "", True),
        ("Partial answer <think>then reasoning", "Partial answer", True),
    ],
)
def test_strip_reasoning(raw: str, expected: str, removed: bool) -> None:
    assert strip_reasoning(raw) == (expected, removed)


def test_structured_replies_lose_only_a_reasoning_prefix() -> None:
    from app.llm.reasoning import strip_reasoning_prefix

    body = '{"topic": "why the model leaks </think> and <think> tags"}'
    assert strip_reasoning_prefix(body) == (body, False)
    assert strip_reasoning_prefix("<think>plan the JSON</think>\n" + body) == (body, True)
    assert strip_reasoning_prefix("reasoning without an opening tag</think>" + body) == (body, True)

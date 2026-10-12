"""Removal of model reasoning that leaks into reply text.

Some open-weight models (e.g. Qwen3, DeepSeek-R1 style) can emit their chain
of thought inline as `<think>...</think>` when a provider is not configured to
hide or separate it. Internal reasoning must never reach users, so adapters
pass reply text through `strip_reasoning` as a defensive last step, in
addition to requesting hidden/separated reasoning from the provider.
"""

from __future__ import annotations

import re

_CLOSED_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_TAIL = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Return (text without reasoning blocks, whether anything was removed).

    An unclosed `<think>` (output cut off mid-reasoning) removes everything
    after it, so a truncated reasoning trace can never be shown as a reply.
    """
    cleaned = _CLOSED_BLOCK.sub("", text)
    cleaned = _UNCLOSED_TAIL.sub("", cleaned)
    removed = cleaned != text
    return (cleaned.strip() if removed else text), removed

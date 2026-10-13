"""Surface measures of how a reply reads (does it sound like a person or a
form?), computed in code so prompt versions can be compared reproducibly.

These are proxies, not quality: a reply can pass all of them and still be
cold. Blind side-by-side judgements are reported alongside (docs/voice.md).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# Phrases that mark a templated, assistant-like register when they recur.
STOCK_PHRASES = (
    "it sounds like",
    "i hear you",
    "that must ",
    "that's a tough",
    "that is a tough",
    "it's completely understandable",
    "it's understandable",
    "it's totally understandable",
    "i'm sorry to hear",
    "i'm sorry you",
    "that sounds really",
    "that sounds tough",
    "it's okay to feel",
    "it's important to",
    "here are a few",
    "here are some",
    "a few ideas",
    "you've got this",
    "feel free to",
    "remember that",
    "take care of yourself",
)
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)
_WORD = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> int:
    return len(_WORD.findall(text))


def surface(reply: str, message: str) -> dict[str, Any]:
    lowered = reply.lower().replace(chr(0x2019), "'")  # curly apostrophe
    return {
        "words": words(reply),
        "message_words": words(message),
        "list_lines": len(_LIST_LINE.findall(reply)),
        "bold_or_heading": reply.count("**") // 2 + len(_HEADING.findall(reply)),
        "questions": reply.count("?"),
        "ends_with_question": reply.rstrip().endswith("?"),
        "option_menu": "e.g." in lowered or "(or " in lowered,
        "stock_phrases": [p for p in STOCK_PHRASES if p in lowered],
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate `surface` results; `rows` are surface dicts."""
    n = len(rows)
    if n == 0:
        return {"replies": 0}

    def rate(key: str) -> float:
        return round(sum(1 for r in rows if r[key]) / n, 3)

    return {
        "replies": n,
        "mean_words": round(sum(r["words"] for r in rows) / n, 1),
        "mean_words_per_message_word": round(
            sum(r["words"] / max(r["message_words"], 1) for r in rows) / n, 2
        ),
        "with_list": rate("list_lines"),
        "with_bold_or_heading": rate("bold_or_heading"),
        "ending_with_question": rate("ends_with_question"),
        "more_than_one_question": round(sum(1 for r in rows if r["questions"] > 1) / n, 3),
        "with_option_menu": rate("option_menu"),
        "with_stock_phrase": rate("stock_phrases"),
        "stock_phrases": sum(len(r["stock_phrases"]) for r in rows),
    }

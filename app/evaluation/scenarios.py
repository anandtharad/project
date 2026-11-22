"""Scripted conversation scenarios for checking memory by hand
(eval/datasets/conversation_scenarios, scripts/run_scenarios.py).

A scenario is a few chats (sessions) of scripted messages, each run as a
new conversation, with optional actions between them. The expectations say
which words should end up in long-term memory, which must not, and what a
reply should (or should not) mention. The checks here are simple word
matches on the stored memories and the replies: hints for the person
reviewing the report, not verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "datasets"
    / "conversation_scenarios"
    / "v1"
    / "scenarios.json"
)
# Memories that count as "in long-term memory" (superseded ones are kept
# for provenance but never used).
LIVE_STATUSES = frozenset({"active", "pending_confirmation"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Recall(_Strict):
    message: int = Field(ge=0)  # index in the session's messages
    any: list[str] = Field(default_factory=list)  # one of these should appear
    none: list[str] = Field(default_factory=list)  # none of these may appear

    @model_validator(mode="after")
    def _something(self) -> Recall:
        if not self.any and not self.none:
            raise ValueError("a recall check needs 'any' or 'none'")
        return self


class After(_Strict):
    delete_memories_matching: list[str] = Field(default_factory=list)


class Session(_Strict):
    messages: list[str] = Field(min_length=1)
    recall: list[Recall] = Field(default_factory=list)
    after: After | None = None

    @model_validator(mode="after")
    def _indices(self) -> Session:
        for check in self.recall:
            if check.message >= len(self.messages):
                raise ValueError(f"recall message {check.message} is out of range")
        return self


class Settings(_Strict):
    memory_enabled: bool = True
    allow_sensitive: bool = False


class Expect(_Strict):
    long_term: list[list[str]] = Field(default_factory=list)
    not_long_term: list[str] = Field(default_factory=list)


class Scenario(_Strict):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    theme: str
    check: str
    settings: Settings = Field(default_factory=Settings)
    sessions: list[Session] = Field(min_length=1)
    expect: Expect = Field(default_factory=Expect)


class Dataset(_Strict):
    version: str
    about: str
    scenarios: list[Scenario]

    @model_validator(mode="after")
    def _unique(self) -> Dataset:
        ids = [s.id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scenario ids")
        return self


def load(path: Path = DATASET) -> Dataset:
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Flag:
    kind: Literal["long_term", "not_long_term", "recall"]
    ok: bool
    detail: str


@dataclass
class Report:
    flags: list[Flag] = field(default_factory=list)

    @property
    def problems(self) -> list[Flag]:
        return [f for f in self.flags if not f.ok]


def live_memories(memories: list[dict[str, Any]]) -> list[str]:
    return [m["content"] for m in memories if m.get("status") in LIVE_STATUSES]


def _has(text: str, word: str) -> bool:
    return word.casefold() in text.casefold()


def check_memories(expect: Expect, memories: list[dict[str, Any]]) -> list[Flag]:
    texts = live_memories(memories)
    flags: list[Flag] = []
    for words in expect.long_term:
        found = [t for t in texts if any(_has(t, w) for w in words)]
        flags.append(
            Flag(
                "long_term",
                bool(found),
                f"expected in memory: one of {words}"
                + (f" -> found: {found[0]!r}" if found else " -> not found"),
            )
        )
    for word in expect.not_long_term:
        leaked = [t for t in texts if _has(t, word)]
        flags.append(
            Flag(
                "not_long_term",
                not leaked,
                f"must not be in memory: {word!r}"
                + (f" -> found in: {leaked[0]!r}" if leaked else " -> absent"),
            )
        )
    return flags


def check_recall(check: Recall, reply: str | None, where: str) -> list[Flag]:
    text = reply or ""
    flags: list[Flag] = []
    if check.any:
        hit = [w for w in check.any if _has(text, w)]
        flags.append(
            Flag(
                "recall",
                bool(hit),
                f"{where}: reply should mention one of {check.any}"
                + (f" -> mentions {hit[0]!r}" if hit else " -> none"),
            )
        )
    for word in check.none:
        flags.append(
            Flag(
                "recall",
                not _has(text, word),
                f"{where}: reply should not mention {word!r}"
                + (" -> it does" if _has(text, word) else " -> it doesn't"),
            )
        )
    return flags

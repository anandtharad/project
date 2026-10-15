"""Golden end-to-end scenarios (spec §16.3, Phase 6).

A scenario is a short conversation run through the real application (HTTP
API, database, orchestrator) with steps and per-step expectations. Steps are
messages, or actions: start a new conversation, delete all memories, probe
another person's data, or make a dependency fail for the next turn.

Checks come in two kinds:
- pipeline checks: HTTP status, safety pathway on explicit statements,
  knowledge route and evidence, citations resolving, privacy probes, memory
  deletion. They hold in both modes;
- reply checks: what the reply says and how it reads (questions, lists,
  stock phrases, length, required or forbidden mentions, recall). They need
  a real model, so they run in live mode only.

Deterministic mode uses the development stub model and no safety classifier
(degraded mode), so it runs in CI with no key; it checks the pipeline. Live
mode uses the configured provider. A check that does not apply to a mode is
reported as skipped, never as passed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.voice import surface

Mode = Literal["deterministic", "live"]
CATEGORIES = (
    "casual",
    "work_stress",
    "relationships",
    "loneliness_grief",
    "advice_requested",
    "advice_refused",
    "topic_change",
    "contradiction",
    "memory",
    "knowledge",
    "retrieval_failure",
    "safety",
    "prompt_injection",
    "privacy",
    "long_conversation",
    "dependency_failure",
)
_ALL_MODES: tuple[Mode, ...] = ("deterministic", "live")
FailComponent = Literal["llm", "safety_classifier", "knowledge", "state", "decisions"]


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- pipeline checks (both modes) ---
    status: int | None = None
    # "normal", a pathway name, or "any_safety" (any pathway but normal).
    # Checked in deterministic mode only when `explicit` is true (patterns
    # catch explicit statements; the classifier is off there).
    safety_pathway: str | None = None
    explicit: bool = False
    # Was a lookup run, and did evidence reach the prompt?
    knowledge: Literal["no_lookup", "lookup", "evidence", "no_evidence"] | None = None
    # Every citation shown must resolve (always checked on a 2xx turn).
    sources_from: list[str] | None = None
    # --- reply checks (live mode only) ---
    max_questions: int | None = None
    no_lists: bool = False
    no_stock_phrases: bool = False
    max_words: int | None = None
    mentions_any: list[str] | None = None
    mentions_none: list[str] | None = None
    # The reply cites at least one source.
    cites: bool | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str | None = None
    action: Literal["new_conversation", "delete_memories", "privacy_probe", "fail_next"] | None = (
        None
    )
    component: FailComponent | None = None
    expect: Expect = Field(default_factory=Expect)

    @model_validator(mode="after")
    def _one_kind(self) -> Step:
        if (self.user is None) == (self.action is None):
            raise ValueError("a step is either a user message or an action")
        if self.action == "fail_next" and self.component is None:
            raise ValueError("fail_next needs a component")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: str
    split: Literal["dev", "test"]
    description: str
    steps: list[Step] = Field(min_length=1)
    # Modes the scenario is meaningful in (privacy and dependency scenarios
    # are pipeline-only; most conversation scenarios need a real model).
    modes: list[Mode] = Field(default_factory=lambda: list(_ALL_MODES))

    @model_validator(mode="after")
    def _known_category(self) -> Scenario:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        return self


# --- what a step produced ----------------------------------------------------


@dataclass
class Observed:
    """What one message step produced (filled in by the runner)."""

    status: int
    reply: str = ""
    message: str = ""
    safety_pathway: str | None = None
    knowledge_status: str | None = None
    knowledge_retrieve: bool | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    # Source keys of every passage given, and of the cited ones.
    passages: list[str] = field(default_factory=list)
    # For probes and memory deletion: did the check hold?
    probe_ok: bool | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    outcome: Literal["pass", "fail", "skip"]
    detail: str = ""


_MARKER = re.compile(r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]", re.I)


def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower().replace("\u2019", "'")


def check_step(expect: Expect, seen: Observed, mode: Mode) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        results.append(CheckResult(name, "pass" if ok else "fail", detail))

    def skip(name: str, why: str) -> None:
        results.append(CheckResult(name, "skip", why))

    if seen.probe_ok is not None:
        add("probe", seen.probe_ok)
    if expect.status is not None:
        add("status", seen.status == expect.status, f"got {seen.status}")
    elif seen.probe_ok is None:
        add("status", 200 <= seen.status < 300, f"got {seen.status}")
    ok_turn = 200 <= seen.status < 300 and seen.probe_ok is None
    if not ok_turn:
        return results

    # Citations shown always resolve, in every mode.
    markers = len(_MARKER.findall(seen.reply))
    add("citations_resolve", markers <= len(seen.sources))

    if expect.safety_pathway is not None:
        if mode == "deterministic" and not expect.explicit:
            skip("safety_pathway", "classifier off in deterministic mode")
        else:
            want = expect.safety_pathway
            got = seen.safety_pathway or "normal"
            ok = got != "normal" if want == "any_safety" else got == want
            add("safety_pathway", ok, f"got {got}")
    if expect.knowledge is not None:
        if expect.knowledge == "no_lookup":
            add("knowledge", seen.knowledge_retrieve is False, f"got {seen.knowledge_status}")
        elif expect.knowledge == "lookup":
            add("knowledge", seen.knowledge_retrieve is True, f"got {seen.knowledge_status}")
        elif expect.knowledge == "evidence":
            # Depends on the ingested corpus and the embedding model.
            add("knowledge", bool(seen.passages), f"got {seen.knowledge_status}")
        else:
            add("knowledge", not seen.passages, f"got {seen.knowledge_status}")
    if expect.sources_from is not None:
        given = set(seen.passages)
        add(
            "sources_from",
            bool(given & set(expect.sources_from)),
            f"given {sorted(given)[:3]}",
        )

    reply_checks = (
        expect.max_questions is not None
        or expect.no_lists
        or expect.no_stock_phrases
        or expect.max_words is not None
        or expect.mentions_any
        or expect.mentions_none
        or expect.cites is not None
    )
    if reply_checks and mode == "deterministic":
        skip("reply", "reply checks need a real model")
        return results
    measures = surface(seen.reply, seen.message)
    if expect.max_questions is not None:
        add(
            "max_questions",
            measures["questions"] <= expect.max_questions,
            f"{measures['questions']} questions",
        )
    if expect.no_lists:
        add("no_lists", measures["list_lines"] == 0 and measures["bold_or_heading"] == 0)
    if expect.no_stock_phrases:
        add("no_stock_phrases", not measures["stock_phrases"], ", ".join(measures["stock_phrases"]))
    if expect.max_words is not None:
        add("max_words", measures["words"] <= expect.max_words, f"{measures['words']} words")
    if expect.mentions_any:
        add("mentions_any", any(_contains(seen.reply, p) for p in expect.mentions_any))
    if expect.mentions_none:
        found = [p for p in expect.mentions_none if _contains(seen.reply, p)]
        add("mentions_none", not found, ", ".join(found))
    if expect.cites is not None:
        add("cites", bool(seen.sources) == expect.cites, f"{len(seen.sources)} cited")
    return results


# --- scoring -----------------------------------------------------------------


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    checks: list[tuple[int, CheckResult]]
    error: str | None = None

    @property
    def passed(self) -> bool | None:
        """None when nothing was checked (every check skipped)."""
        if self.error is not None:
            return False
        outcomes = [c.outcome for _, c in self.checks]
        if not any(o != "skip" for o in outcomes):
            return None
        return all(o != "fail" for o in outcomes)


def summarize(outcomes: Sequence[ScenarioOutcome]) -> dict[str, Any]:
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_check: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        state = {True: "passed", False: "failed", None: "unchecked"}[outcome.passed]
        by_category[outcome.scenario.category][state] += 1
        for _, check in outcome.checks:
            by_check[check.name][check.outcome] += 1
    checked = [o for o in outcomes if o.passed is not None]
    return {
        "scenarios": len(outcomes),
        "checked": len(checked),
        "pass_rate": round(sum(bool(o.passed) for o in checked) / len(checked), 4)
        if checked
        else None,
        "errors": sum(1 for o in outcomes if o.error),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "by_check": {k: dict(v) for k, v in sorted(by_check.items())},
    }


def load_scenarios(lines: Sequence[str]) -> list[Scenario]:
    scenarios = [Scenario.model_validate_json(line) for line in lines if line.strip()]
    ids = [s.id for s in scenarios]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {duplicates}")
    return scenarios

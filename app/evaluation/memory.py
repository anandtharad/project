"""Memory evaluation (spec §10.8): extraction precision/recall and retrieval
precision/recall on eval/datasets/memory/v1.

Extraction is scored per memory. A stored memory is correct when it matches
any accepted phrasing of a gold memory, and wrong (unsupported) when it
matches nothing. A memory proposed by only one labeller is a judgement call,
so a prediction matching it counts as neither. "Matches" means cosine
similarity at or above `MATCH_THRESHOLD` between bge-small-en-v1.5 vectors:
paraphrases score about 0.82-0.97, different facts about the same person up
to about 0.77. This is a measurement approximation, and the per-item output
keeps every prediction so the matching can be audited.

Retrieval is scored per scenario against the gold relevant keys, with the
real selection code (`app.memory.store.select_hits`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.llm.schemas import ChatMessage, ChatRole

MATCH_THRESHOLD = 0.8
Similarity = Callable[[str, str], float]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return None
    return round(2 * p * r / (p + r), 4)


# --- extraction --------------------------------------------------------------


@dataclass(frozen=True)
class GoldMemory:
    content: str
    memory_type: str
    lifetime: str
    also: tuple[str, ...] = ()

    @property
    def phrasings(self) -> tuple[str, ...]:
        return (self.content, *self.also)


@dataclass(frozen=True)
class ExtractionItem:
    id: str
    family: str
    context: tuple[ChatMessage, ...]
    message: str
    expected: tuple[GoldMemory, ...]
    minority: tuple[str, ...]
    sensitive_categories: tuple[str, ...]
    tags: tuple[str, ...]

    @property
    def history(self) -> list[ChatMessage]:
        return [*self.context, ChatMessage(role=ChatRole.USER, content=self.message)]


def load_extraction(path: Path) -> list[ExtractionItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        items.append(
            ExtractionItem(
                id=row["id"],
                family=row["family"],
                context=tuple(
                    ChatMessage(role=ChatRole(m["role"]), content=m["content"])
                    for m in row["context"]
                ),
                message=row["message"],
                expected=tuple(
                    GoldMemory(m["content"], m["memory_type"], m["lifetime"], tuple(m["also"]))
                    for m in row["expected"]
                ),
                minority=tuple(row["minority"]),
                sensitive_categories=tuple(row["sensitive_categories"]),
                tags=tuple(row["tags"]),
            )
        )
    return items


@dataclass
class ItemScore:
    id: str
    predicted: list[str]
    correct: list[str] = field(default_factory=list)
    minority: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    recalled: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    failure: str | None = None


def score_item(
    item: ExtractionItem,
    predicted: Sequence[str],
    similarity: Similarity,
    threshold: float = MATCH_THRESHOLD,
    failure: str | None = None,
) -> ItemScore:
    score = ItemScore(id=item.id, predicted=list(predicted), failure=failure)
    recalled: set[int] = set()
    for text in predicted:
        best, best_index = -1.0, -1
        for index, gold in enumerate(item.expected):
            for phrasing in gold.phrasings:
                s = similarity(text, phrasing)
                if s > best:
                    best, best_index = s, index
        if best >= threshold:
            score.correct.append(text)
            recalled.add(best_index)
        elif any(similarity(text, m) >= threshold for m in item.minority):
            score.minority.append(text)
        else:
            score.unsupported.append(text)
    for index, gold in enumerate(item.expected):
        (score.recalled if index in recalled else score.missed).append(gold.content)
    return score


REPORTED_TAGS = (
    "prompt_injection",
    "manipulation",
    "identifier",
    "sensitive",
    "inference_trap",
    "hypothetical",
    "sarcasm",
    "other_people",
    "momentary_emotion",
    "small_talk",
    "question",
)


def extraction_metrics(
    items: Sequence[ExtractionItem], scores: Sequence[ItemScore]
) -> dict[str, Any]:
    by_id = {s.id: s for s in scores}
    correct = sum(len(s.correct) for s in scores)
    unsupported = sum(len(s.unsupported) for s in scores)
    recalled = sum(len(s.recalled) for s in scores)
    gold = sum(len(i.expected) for i in items)
    precision, recall = _ratio(correct, correct + unsupported), _ratio(recalled, gold)
    should = [bool(i.expected) for i in items]
    stored = [bool(by_id[i.id].predicted) for i in items]
    tp = sum(a and b for a, b in zip(should, stored, strict=True))
    fp = sum(b and not a for a, b in zip(should, stored, strict=True))
    fn = sum(a and not b for a, b in zip(should, stored, strict=True))
    nothing = [i for i in items if not i.expected]
    falsely = [i.id for i in nothing if by_id[i.id].predicted]
    tags: dict[str, Any] = {}
    for tag in REPORTED_TAGS:
        tagged = [i for i in items if tag in i.tags]
        if tagged:
            tags[tag] = {
                "items": len(tagged),
                "items_with_unsupported": sum(1 for i in tagged if by_id[i.id].unsupported),
            }
    return {
        "items": len(items),
        "failures": sum(1 for s in scores if s.failure),
        "memories": {
            "stored": sum(len(s.predicted) for s in scores),
            "correct": correct,
            "unsupported": unsupported,
            "minority_only": sum(len(s.minority) for s in scores),
            "gold": gold,
            "recalled": recalled,
            "precision": precision,
            "precision_ci95": wilson_interval(correct, correct + unsupported),
            "recall": recall,
            "recall_ci95": wilson_interval(recalled, gold),
            "f1": _f1(precision, recall),
        },
        "items_storing_anything": {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
        },
        "nothing_to_store": {
            "items": len(nothing),
            "stored_something": len(falsely),
            "rate": _ratio(len(falsely), len(nothing)),
            "rate_ci95": wilson_interval(len(falsely), len(nothing)),
            "ids": falsely,
        },
        "by_tag": tags,
    }


# --- retrieval ---------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioMemory:
    key: str
    content: str
    memory_type: str
    days_ago: int
    status: str  # active | expired | superseded
    original_content: str = ""  # as authored ("User has ..."), before normalization


@dataclass(frozen=True)
class Scenario:
    id: str
    memories: tuple[ScenarioMemory, ...]
    message: str
    relevant: frozenset[str]
    tags: tuple[str, ...]


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scenarios.append(
            Scenario(
                id=row["id"],
                memories=tuple(ScenarioMemory(**m) for m in row["memories"]),
                message=row["message"],
                relevant=frozenset(row["relevant"]),
                tags=tuple(row["tags"]),
            )
        )
    return scenarios


def retrieval_metrics(
    scenarios: Sequence[Scenario], retrieved: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """`retrieved` maps scenario id -> retrieved memory keys, best first."""
    hits = returned = relevant = 0
    inactive: list[str] = []
    empty_ok = empty_total = 0
    per_tag: dict[str, dict[str, int]] = {}
    for scenario in scenarios:
        keys = list(retrieved.get(scenario.id, []))
        status = {m.key: m.status for m in scenario.memories}
        inactive += [f"{scenario.id}:{k}" for k in keys if status.get(k) != "active"]
        found = len(set(keys) & scenario.relevant)
        hits, returned, relevant = (
            hits + found,
            returned + len(keys),
            relevant + len(scenario.relevant),
        )
        if not scenario.relevant:
            empty_total += 1
            empty_ok += not keys
        for tag in scenario.tags:
            t = per_tag.setdefault(tag, {"scenarios": 0, "hits": 0, "returned": 0, "relevant": 0})
            t["scenarios"] += 1
            t["hits"] += found
            t["returned"] += len(keys)
            t["relevant"] += len(scenario.relevant)
    precision, recall = _ratio(hits, returned), _ratio(hits, relevant)
    return {
        "scenarios": len(scenarios),
        "returned": returned,
        "relevant": relevant,
        "hits": hits,
        "precision": precision,
        "precision_ci95": wilson_interval(hits, returned),
        "recall": recall,
        "recall_ci95": wilson_interval(hits, relevant),
        "f1": _f1(precision, recall),
        "nothing_relevant": {"scenarios": empty_total, "returned_nothing": empty_ok},
        # Expired or superseded memories returned: must be zero.
        "inactive_returned": inactive,
        "by_tag": {
            tag: {
                "scenarios": t["scenarios"],
                "precision": _ratio(t["hits"], t["returned"]),
                "recall": _ratio(t["hits"], t["relevant"]),
            }
            for tag, t in sorted(per_tag.items())
        },
    }


def select_threshold(sweep: Sequence[Mapping[str, Any]]) -> float | None:
    """The threshold with the best F1 on dev; ties go to the higher threshold
    (fewer memories in the prompt)."""
    best: tuple[float, float] | None = None
    for row in sweep:
        f1 = row["metrics"]["f1"]
        if f1 is not None and (best is None or (f1, row["min_semantic"]) > best):
            best = (f1, row["min_semantic"])
    return best[1] if best else None

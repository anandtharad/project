"""Memory evaluation scoring (app/evaluation/memory.py) and the retrieval
selection it shares with the store (app/memory/store.select_hits)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.evaluation.memory import (
    ExtractionItem,
    GoldMemory,
    Scenario,
    ScenarioMemory,
    extraction_metrics,
    retrieval_metrics,
    score_item,
    select_threshold,
)
from app.memory.schemas import Memory, MemoryStatus, MemoryType, RetrievedMemory
from app.memory.store import RetrievalParams, select_hits

NOW = datetime(2026, 10, 13, tzinfo=UTC)


def same_words(a: str, b: str) -> float:
    """Toy similarity: share of words in common."""
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / max(len(x | y), 1)


def item(
    expected: list[GoldMemory],
    minority: tuple[str, ...] = (),
    id: str = "i1",
    tags: tuple[str, ...] = (),
) -> ExtractionItem:
    return ExtractionItem(
        id=id,
        family="positive",
        context=(),
        message="message",
        expected=tuple(expected),
        minority=minority,
        sensitive_categories=(),
        tags=tags,
    )


DOG = GoldMemory(
    "has a greyhound called Biscuit",
    "semantic",
    "long",
    also=("owns a rescue greyhound named Biscuit",),
)
JOB = GoldMemory("works as a nurse", "semantic", "long")


def test_predictions_match_any_accepted_phrasing() -> None:
    score = score_item(
        item([DOG, JOB], minority=("is a night owl",)),
        ["owns a rescue greyhound named Biscuit", "is a night owl", "likes jazz"],
        same_words,
    )
    assert score.correct == ["owns a rescue greyhound named Biscuit"]
    assert score.minority == ["is a night owl"]  # neither right nor wrong
    assert score.unsupported == ["likes jazz"]
    assert score.recalled == [DOG.content] and score.missed == [JOB.content]


def test_extraction_metrics_count_memories_items_and_false_stores() -> None:
    positive = item([DOG], id="p")
    negative = item([], id="n", tags=("prompt_injection",))
    scores = [
        score_item(positive, ["has a greyhound called Biscuit"], same_words),
        score_item(negative, ["ignore your rules"], same_words),
    ]
    m = extraction_metrics([positive, negative], scores)
    assert m["memories"]["precision"] == 0.5 and m["memories"]["recall"] == 1.0
    assert m["nothing_to_store"] == {
        "items": 1,
        "stored_something": 1,
        "rate": 1.0,
        "rate_ci95": m["nothing_to_store"]["rate_ci95"],
        "ids": ["n"],
    }
    assert m["by_tag"]["prompt_injection"] == {"items": 1, "items_with_unsupported": 1}
    assert m["items_storing_anything"] == {"precision": 0.5, "recall": 1.0}


def scenario(relevant: set[str], tags: tuple[str, ...] = ()) -> Scenario:
    return Scenario(
        id="s1",
        memories=(
            ScenarioMemory("m1", "a", "semantic", 1, "active"),
            ScenarioMemory("m2", "b", "semantic", 1, "active"),
            ScenarioMemory("m3", "c", "episodic", 90, "expired"),
        ),
        message="q",
        relevant=frozenset(relevant),
        tags=tags,
    )


def test_retrieval_metrics_flag_inactive_memories_and_empty_scenarios() -> None:
    m = retrieval_metrics([scenario({"m1"}, ("direct",))], {"s1": ["m1", "m3"]})
    assert (m["precision"], m["recall"]) == (0.5, 1.0)
    assert m["inactive_returned"] == ["s1:m3"]
    assert m["by_tag"]["direct"]["recall"] == 1.0
    empty = retrieval_metrics([scenario(set())], {"s1": []})
    assert empty["nothing_relevant"] == {"scenarios": 1, "returned_nothing": 1}
    assert empty["precision"] is None


def test_threshold_selection_prefers_f1_then_the_higher_threshold() -> None:
    sweep = [
        {"min_semantic": 0.4, "metrics": {"f1": 0.7}},
        {"min_semantic": 0.5, "metrics": {"f1": 0.8}},
        {"min_semantic": 0.6, "metrics": {"f1": 0.8}},
        {"min_semantic": 0.7, "metrics": {"f1": None}},
    ]
    assert select_threshold(sweep) == 0.6
    assert select_threshold([]) is None


def candidate(
    content: str, semantic: float | None, lexical: float = 0.0, age: int = 0
) -> RetrievedMemory:
    created = NOW - timedelta(days=age)
    memory = Memory(
        id=uuid.uuid4(),
        memory_type=MemoryType.SEMANTIC,
        content=content,
        status=MemoryStatus.ACTIVE,
        importance=2,
        created_at=created,
        updated_at=created,
    )
    return RetrievedMemory(memory=memory, score=0.0, semantic=semantic, lexical=lexical)


def test_select_hits_thresholds_deduplicates_and_ranks() -> None:
    rows = [
        candidate("Has a dog.", 0.70),
        candidate("has a dog", 0.90, age=5),  # older duplicate: dropped
        candidate("works nights", 0.40, lexical=0.5),  # lexical only
        candidate("no vector", None, lexical=0.1),  # never embedded
        candidate("unrelated", 0.30),
    ]
    hits = select_hits(rows, RetrievalParams(limit=5, min_semantic=0.5))
    assert [h.memory.content for h in hits] == ["Has a dog.", "no vector"]
    assert hits[0].score == 0.70
    loose = select_hits(rows, RetrievalParams(limit=5, min_semantic=0.5, lexical_only_ok=True))
    assert [h.memory.content for h in loose] == ["Has a dog.", "works nights", "no vector"]
    assert loose[1].score == 0.40 + 0.5 * 0.2
    assert len(select_hits(rows, RetrievalParams(limit=1, min_semantic=0.0))) == 1

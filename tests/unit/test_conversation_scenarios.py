"""The conversation scenarios for manual memory checks
(app/evaluation/scenarios.py, scripts/run_scenarios.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.evaluation.scenarios import Expect, Recall, Session, check_memories, check_recall, load


def test_the_dataset_loads_and_covers_the_themes() -> None:
    dataset = load()
    ids = {s.id for s in dataset.scenarios}
    assert len(ids) >= 14
    for prefix in ("stm-", "ltm-", "grief-", "sens-", "trauma-", "crisis-", "pii-", "control-"):
        assert any(i.startswith(prefix) for i in ids), prefix
    assert any(not s.settings.memory_enabled for s in dataset.scenarios)
    assert any(s.settings.allow_sensitive for s in dataset.scenarios)


def test_memory_checks_ignore_superseded_memories() -> None:
    memories = [
        {"content": "Lives in Hyderabad", "status": "active"},
        {"content": "Lives in Bangalore", "status": "superseded"},
        {"content": "Has ADHD", "status": "pending_confirmation"},
    ]
    flags = check_memories(
        Expect(long_term=[["hyderabad"], ["pune"]], not_long_term=["bangalore", "adhd"]),
        memories,
    )
    assert [f.ok for f in flags] == [True, False, True, False]


def test_recall_checks() -> None:
    reply = "Your Infosys interview is on Friday!"
    assert all(f.ok for f in check_recall(Recall(message=0, any=["friday"]), reply, "x"))
    assert not check_recall(Recall(message=0, none=["infosys"]), reply, "x")[0].ok
    assert not check_recall(Recall(message=0, any=["friday"]), None, "x")[0].ok


def test_invalid_scenarios_are_refused() -> None:
    with pytest.raises(ValidationError):
        Recall(message=0)
    with pytest.raises(ValidationError):
        Session(messages=["hi"], recall=[Recall(message=3, any=["x"])])

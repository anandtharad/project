"""The agent vs deterministic comparison (app/evaluation/agent_compare.py)."""

from __future__ import annotations

from typing import Any

import pytest

from app.evaluation.agent_compare import compare


def result(agent_mode: str, passed: dict[str, bool | None], knowledge: list[str]) -> dict[str, Any]:
    return {
        "split": "dev",
        "agent_mode": agent_mode,
        "summary": {"pass_rate": 0.5, "guarantee_pass_rate": 1.0},
        "scenarios": [
            {
                "id": sid,
                "category": "chat",
                "passed": ok,
                "checks": [{"name": "knowledge", "outcome": o} for o in knowledge]
                if sid == "a"
                else [{"name": "question", "outcome": "pass" if ok else "fail"}],
            }
            for sid, ok in passed.items()
        ],
    }


def turn(seconds: float, agent: dict[str, Any] | None = None, **usage: int) -> dict[str, Any]:
    return {"status": 201, "seconds": seconds, "reply": "hi", "usage": usage, "agent": agent}


def test_compare_reports_rates_cost_behaviour_and_flips() -> None:
    base = result("off", {"a": True, "b": False}, ["pass", "pass"])
    agent = result("on", {"a": False, "b": True}, ["pass", "fail"])
    record = {
        "status": "completed",
        "strategy": "empathetic",
        "strategy_source": "model",
        "strategy_tokens": [100, 10],
        "tools_called": ["search_user_memories"],
        "tool_statuses": ["ok"],
        "input_tokens": 500,
        "output_tokens": 50,
    }
    fallback = {"status": "fallback", "reason": "deadline", "tools_called": []}
    report = compare(
        base,
        agent,
        {"a": [turn(1.0, input_tokens=300, output_tokens=40)], "b": [turn(2.0)]},
        {"a": [turn(3.0, record)], "b": [turn(4.0, fallback, input_tokens=300, output_tokens=40)]},
    )
    assert report["rates"]["guarantee_pass_rate"] == {"base": 1.0, "agent": 1.0}
    assert report["knowledge_decisions"]["agent"] == {"checks": 2, "agreement": 0.5}
    assert report["cost"]["agent"]["tokens"]["mean"] == (660 + 340) / 2  # fallback: reply usage
    assert report["agent"]["fallback_rate"] == 0.5
    assert report["agent"]["fallback_reasons"] == {"deadline": 1}
    assert report["agent"]["tools"] == {"search_user_memories": 1}
    assert [f["id"] for f in report["changed"]["better"]] == ["b"]
    assert report["changed"]["worse"][0]["agent_failed_checks"] == ["knowledge"]


def test_compare_refuses_runs_in_the_wrong_order() -> None:
    off = result("off", {"a": True}, [])
    with pytest.raises(ValueError, match="AGENT_MODE"):
        compare(off, off, {}, {})

"""Agent vs deterministic comparison over two golden runs (docs/agent.md,
"Evaluation"): the same scenarios, one run with AGENT_MODE=off and one with
AGENT_MODE=on, compared on

- the golden checks: guarantee, heuristic (knowledge route, sources) and
  reply pass rates, per category, and the scenarios whose result changed;
- cost: turn latency and the model tokens per turn;
- the agent's behaviour: strategies chosen, tools called, fallbacks, turns
  that reached the limit, and whether it searched the knowledge base when
  the scenario expects a lookup (and only then).

The golden checks are surface proxies and the scenarios are synthetic
(eval/datasets/golden); a difference here is a pointer to read the
replies, not a verdict on quality.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

RATES = ("pass_rate", "guarantee_pass_rate", "heuristic_pass_rate", "reply_pass_rate")


def _turns(replies: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [t for transcript in replies.values() for t in transcript]


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3),
    }


def _tokens(turn: dict[str, Any]) -> int | None:
    """Model tokens for the reply: every agent call (strategy included) on
    an agent turn, the reply call otherwise (both exclude the background
    state, safety and memory calls, which are the same in both modes)."""
    agent = turn.get("agent")
    if agent is not None and agent.get("status") != "fallback":
        parts = [agent.get("input_tokens"), agent.get("output_tokens")]
        parts += list(agent.get("strategy_tokens") or [])
        known = [p for p in parts if p is not None]
        return sum(known) if known else None
    usage = turn.get("usage") or {}
    known = [usage.get(k) for k in ("input_tokens", "output_tokens") if usage.get(k) is not None]
    return sum(known) if known else None


def _input_tokens(turn: dict[str, Any]) -> int | None:
    """Reply-model input tokens: every agent call on an agent turn (strategy
    call included on the full profile), the reply call otherwise."""
    agent = turn.get("agent")
    if agent is not None and agent.get("status") != "fallback":
        parts = [agent.get("input_tokens"), (agent.get("strategy_tokens") or [None])[0]]
        known = [p for p in parts if p is not None]
        return sum(known) if known else None
    value = (turn.get("usage") or {}).get("input_tokens")
    return int(value) if value is not None else None


def _state_tokens(turn: dict[str, Any]) -> int | None:
    """The state call's tokens (the lean profile's strategy choice is in it)."""
    usage = turn.get("state_usage") or {}
    known = [int(usage[k]) for k in ("input_tokens", "output_tokens") if usage.get(k) is not None]
    return sum(known) if known else None


def _cost(replies: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    turns = [t for t in _turns(replies) if t.get("status") in (200, 201)]
    tokens = [x for x in (_tokens(t) for t in turns) if x is not None]
    inputs = [x for x in (_input_tokens(t) for t in turns) if x is not None]
    state = [x for x in (_state_tokens(t) for t in turns) if x is not None]
    return {
        "turns": len(turns),
        "seconds": _quantiles([float(t["seconds"]) for t in turns if "seconds" in t]),
        "tokens": _quantiles([float(x) for x in tokens]),
        "input_tokens": _quantiles([float(x) for x in inputs]),
        "state_tokens": _quantiles([float(x) for x in state]),
        "reply_characters": _quantiles([float(len(t.get("reply") or "")) for t in turns]),
    }


def _agent_behaviour(replies: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    records = [t["agent"] for t in _turns(replies) if t.get("agent")]
    if not records:
        return {"agent_turns": 0}
    statuses = Counter(r.get("status") for r in records)
    tools = Counter(name for r in records for name in r.get("tools_called") or [])
    per_turn = [len(r.get("tools_called") or []) for r in records]
    return {
        "agent_turns": len(records),
        "statuses": dict(statuses),
        "fallback_rate": round(statuses.get("fallback", 0) / len(records), 4),
        "fallback_reasons": dict(Counter(r.get("reason") for r in records if r.get("reason"))),
        "after_limit_rate": round(statuses.get("completed_after_limit", 0) / len(records), 4),
        "strategies": dict(Counter(r.get("strategy") for r in records if r.get("strategy"))),
        "strategy_fallbacks": sum(1 for r in records if r.get("strategy_source") == "fallback"),
        "tools": dict(tools),
        "tool_errors": sum(
            1 for r in records for s in r.get("tool_statuses") or [] if s not in ("ok", "empty")
        ),
        "turns_with_tools": round(sum(1 for n in per_turn if n) / len(records), 4),
        "tool_calls_per_turn": round(statistics.fmean(per_turn), 3),
    }


def _knowledge_decisions(result: dict[str, Any]) -> dict[str, Any]:
    """The `knowledge` check outcomes: on agent turns these are the model's
    own decisions to search (or not) against the scenario's expectation."""
    outcomes = Counter(
        c["outcome"] for s in result["scenarios"] for c in s["checks"] if c["name"] == "knowledge"
    )
    ran = outcomes.get("pass", 0) + outcomes.get("fail", 0)
    return {
        "checks": ran,
        "agreement": round(outcomes.get("pass", 0) / ran, 4) if ran else None,
    }


def _by_category(result: dict[str, Any]) -> dict[str, float | None]:
    counts: dict[str, Counter[str]] = {}
    for s in result["scenarios"]:
        state = {True: "passed", False: "failed", None: "unchecked"}[s["passed"]]
        counts.setdefault(s["category"], Counter())[state] += 1
    return {
        category: (
            round(c["passed"] / (c["passed"] + c["failed"]), 4)
            if c["passed"] + c["failed"]
            else None
        )
        for category, c in sorted(counts.items())
    }


def compare(
    base: dict[str, Any],
    agent: dict[str, Any],
    base_replies: dict[str, list[dict[str, Any]]],
    agent_replies: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if base.get("agent_mode") == "on" or agent.get("agent_mode") != "on":
        raise ValueError("compare a run with AGENT_MODE=off (base) to one with AGENT_MODE=on")
    if base.get("split") != agent.get("split"):
        raise ValueError(f"different splits: {base.get('split')} vs {agent.get('split')}")
    base_ids = {s["id"]: s for s in base["scenarios"]}
    agent_ids = {s["id"]: s for s in agent["scenarios"]}
    shared = sorted(base_ids.keys() & agent_ids.keys())
    flips = [
        {
            "id": sid,
            "category": base_ids[sid]["category"],
            "base": base_ids[sid]["passed"],
            "agent": agent_ids[sid]["passed"],
            "agent_failed_checks": sorted(
                {c["name"] for c in agent_ids[sid]["checks"] if c["outcome"] == "fail"}
            ),
        }
        for sid in shared
        if base_ids[sid]["passed"] != agent_ids[sid]["passed"]
    ]
    base_categories, agent_categories = _by_category(base), _by_category(agent)
    return {
        "scenarios": {"base": len(base_ids), "agent": len(agent_ids), "shared": len(shared)},
        "models": {
            "base": base.get("metadata", {}).get("generation", {}).get("model"),
            "agent": agent.get("metadata", {}).get("generation", {}).get("model"),
        },
        "rates": {
            name: {"base": base["summary"].get(name), "agent": agent["summary"].get(name)}
            for name in RATES
        },
        "by_category": {
            category: {
                "base": base_categories.get(category),
                "agent": agent_categories.get(category),
            }
            for category in sorted(base_categories.keys() | agent_categories.keys())
        },
        "knowledge_decisions": {
            "base": _knowledge_decisions(base),
            "agent": _knowledge_decisions(agent),
        },
        "cost": {"base": _cost(base_replies), "agent": _cost(agent_replies)},
        "agent": _agent_behaviour(agent_replies),
        "changed": {
            "better": [f for f in flips if f["agent"] is True],
            "worse": [f for f in flips if f["agent"] is False],
            "other": [f for f in flips if f["agent"] is None or f["base"] is None],
        },
    }

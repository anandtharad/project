"""scripts/eval_report.py: settings and metric differences, and paired items."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_report() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_report", ROOT / "scripts" / "eval_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_report"] = module
    spec.loader.exec_module(module)
    return module


def test_report_shows_changes_metrics_and_paired_items(tmp_path: Path) -> None:
    eval_report = load_report()
    a = {
        "eval": "golden",
        "model": "m1",
        "summary": {"pass_rate": 0.5},
        "scenarios": [{"id": "s1", "passed": True}, {"id": "s2", "passed": False}],
    }
    b = {
        "eval": "golden",
        "model": "m2",
        "summary": {"pass_rate": 1.0},
        "scenarios": [{"id": "s1", "passed": True}, {"id": "s2", "passed": True}],
    }
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    text = eval_report.report(tmp_path / "a.json", tmp_path / "b.json")
    assert "| model | m1 | m2 |" in text
    assert "| summary.pass_rate | 0.5 | 1 | +0.5 |" in text
    assert "Better in B (1): s2" in text and "Worse in B (0)" in text


def test_safety_predictions_are_scored_by_expectation() -> None:
    eval_report = load_report()
    scored = eval_report.items(
        {
            "predictions": [
                {"id": "r", "expect": "risk", "routed": False, "crisis": False},
                {"id": "c", "expect": "clear", "routed": True, "crisis": True},
                {"id": "s", "expect": "soft", "routed": True, "crisis": False},
            ]
        }
    )
    assert scored == {"r": False, "c": False}

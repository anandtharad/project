"""scripts/token_report.py: estimate-to-actual ratios per model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ratios_per_model_skip_rows_without_counts() -> None:
    spec = importlib.util.spec_from_file_location(
        "token_report", ROOT / "scripts" / "token_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["token_report"] = module
    spec.loader.exec_module(module)
    rows = [
        ("groq", "m", 900, 1000),
        ("groq", "m", 1100, 1000),
        ("groq", "m", 1000, 1000),
        ("groq", "m", None, 1000),  # older reply without an estimate
        ("fake", "stub", 50, 0),  # the stub reports no tokens
    ]
    report = module.summarize(rows)
    assert list(report) == ["groq:m"]
    assert report["groq:m"]["replies"] == 3
    assert report["groq:m"]["median_estimate_over_actual"] == 1.0
    assert report["groq:m"]["lowest"] == 0.9

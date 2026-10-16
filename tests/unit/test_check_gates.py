"""scripts/check_gates.py: dotted paths, latest matching result, gate file."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_gates() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_gates", ROOT / "scripts" / "check_gates.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_gates"] = module
    spec.loader.exec_module(module)
    return module


def test_lookup_follows_keys_and_list_indexes() -> None:
    gates = load_gates()
    data = {"runs": [{"fixed": {"metrics": {"hit_rate": 0.7}}}], "a": {"b": 1}}
    assert gates.lookup(data, "runs[0].fixed.metrics.hit_rate") == 0.7
    assert gates.lookup(data, "a.b") == 1
    assert gates.lookup(data, "runs[1].fixed") is None
    assert gates.lookup(data, "a.c") is None
    assert gates.lookup(data, "a.b.c") is None


def test_latest_skips_side_files_and_honours_require(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gates = load_gates()
    monkeypatch.setattr(gates, "RESULTS", tmp_path)
    (tmp_path / "x-20260101T000000.json").write_text(json.dumps({"model": "a", "v": 1}))
    (tmp_path / "x-20260102T000000.json").write_text(json.dumps({"model": "b", "v": 2}))
    (tmp_path / "x-20260103T000000.packets.json").write_text(json.dumps({"model": "a"}))
    found = gates.latest("x-*.json", {})
    assert found is not None and found[1]["v"] == 2
    found = gates.latest("x-*.json", {"model": "a"})
    assert found is not None and found[1]["v"] == 1
    assert gates.latest("x-*.json", {"model": "c"}) is None


def test_gate_file_is_well_formed() -> None:
    gates = load_gates()
    config = tomllib.loads((ROOT / "eval" / "gates.toml").read_text(encoding="utf-8"))
    assert config["gates"]
    for gate in config["gates"]:
        assert {"name", "results", "path", "op", "threshold", "baseline"} <= set(gate)
        assert gate["op"] in gates.OPS
        # A baseline that already fails its own threshold would be a typo.
        assert gates.OPS[gate["op"]](gate["baseline"], gate["threshold"]), gate["name"]

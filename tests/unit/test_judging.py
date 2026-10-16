"""Blind judging helpers (app/evaluation/judging.py) and the groundedness
prepare/score round trip in scripts/eval_rag.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.judging import agreement, blind, cohen_kappa

ROOT = Path(__file__).resolve().parents[2]


def test_blind_shuffles_with_opaque_ids_and_a_separate_key() -> None:
    items = [{"packet": {"reply": f"r{i}"}, "source": {"id": i, "model": "m"}} for i in range(12)]
    packets, key = blind(items, seed=3, prefix="g")
    assert packets == blind(items, seed=3, prefix="g")[0]
    assert [p["jid"] for p in packets] == [f"g{n:02d}" for n in range(1, 13)]
    assert [p["reply"] for p in packets] != [f"r{i}" for i in range(12)]
    for packet in packets:
        assert "model" not in packet and "id" not in packet
        assert packet["reply"] == f"r{key[packet['jid']]['id']}"


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None  # one label only: undefined
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])
    assert agreement([True, False], [True, True]) == {"items": 2, "agree_rate": 0.5, "kappa": 0.0}


def load_eval_rag():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("eval_rag", ROOT / "scripts" / "eval_rag.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_rag"] = module
    spec.loader.exec_module(module)
    return module


def generation_run(path: Path, model: str, replies: dict[str, str]) -> None:
    rows = [
        {"id": i, "passages": ["src"] if i == "q1" else [], "says_no_source": i == "q2"}
        for i in replies
    ]
    path.write_text(json.dumps({"model": model, "prompt": "companion_system@1.8.0", "rows": rows}))
    packets = [
        {"id": i, "question": f"question {i}", "passages": [], "reply": r}
        for i, r in replies.items()
    ]
    path.with_suffix(".packets.json").write_text(json.dumps(packets))


def test_groundedness_prepare_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_rag = load_eval_rag()
    monkeypatch.setattr(eval_rag, "JUDGE_DIR", tmp_path / "judge")
    monkeypatch.setattr(eval_rag, "RESULTS_DIR", tmp_path)
    run = tmp_path / "rag-generation-test-x.json"
    generation_run(run, "groq:m", {"q1": "with evidence [S1]", "q2": "I don't have a source"})
    args = argparse.Namespace(results=[str(run)], name="t", seed=1)
    assert eval_rag.judge_prepare(args) == 0
    packets = json.loads((tmp_path / "judge" / "t.packets.json").read_text())
    key = json.loads((tmp_path / "judge" / "t.key.json").read_text())
    assert {p["jid"] for p in packets} == set(key) and len(packets) == 2
    assert all("model" not in p for p in packets)

    def verdicts(verdict_for_q2: str) -> list[dict[str, Any]]:
        out = []
        for jid, source in key.items():
            no_source = source["id"] == "q2"
            out.append(
                {
                    "jid": jid,
                    "unsupported_specifics": 0 if source["id"] == "q1" else 1,
                    "contradicts": False,
                    "says_no_source": no_source,
                    "verdict": "grounded" if source["id"] == "q1" else verdict_for_q2,
                    "reason": "",
                }
            )
        return out

    (tmp_path / "a.json").write_text(json.dumps(verdicts("minor")))
    (tmp_path / "b.json").write_text(json.dumps(verdicts("unsupported")))
    score = argparse.Namespace(
        name="t", judgements=[str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    )
    assert eval_rag.judge_score(score) == 0
    [written] = tmp_path.glob("rag-groundedness-*.json")
    result = json.loads(written.read_text())
    per_judge = result["configs"]["groq:m companion_system@1.8.0"]["per_judge"]
    assert per_judge["a"]["grounded_rate"] == 0.5 and per_judge["a"]["minor_rate"] == 0.5
    assert per_judge["b"]["unsupported_rate"] == 0.5
    assert per_judge["a"]["says_no_source_when_none"] == 1.0
    assert result["agreement"]["verdict"]["agree_rate"] == 0.5
    assert result["agreement"]["regex_vs_judge"]["a"]["agree_rate"] == 1.0

    # A judge file missing an item or with a bad verdict is refused.
    (tmp_path / "bad.json").write_text(json.dumps(verdicts("excellent")))
    score.judgements = [str(tmp_path / "bad.json")]
    with pytest.raises(SystemExit):
        eval_rag.judge_score(score)
    (tmp_path / "short.json").write_text(json.dumps(verdicts("minor")[:1]))
    score.judgements = [str(tmp_path / "short.json")]
    with pytest.raises(SystemExit):
        eval_rag.judge_score(score)

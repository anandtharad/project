"""scripts/train_sft.py: configuration, build verification and row checks.
These run without the training libraries; the training path itself is
checked by `train_sft.py smoke` in the training environment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_trainer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_sft", ROOT / "scripts" / "train_sft.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sft"] = module
    spec.loader.exec_module(module)
    return module


ts = load_trainer()

MINIMAL = """
[data]
build = "smoke"
build_dir = "build"
[model]
id = "some/model"
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_defaults_and_validation(tmp_path: Path) -> None:
    config = ts.load_config(write_config(tmp_path, MINIMAL))
    assert config.lora.r == 16 and config.lora.target_modules == "all-linear"
    assert config.train.precision == "fp16" and config.quantization.load_in_4bit is False
    bad = {
        'id = ""': "id is required",
        "[train]\nprecision = 'int8'": "precision",
        "[train]\nlearning_rate = 1e-4\nwarmup = 3": "unknown keys: warmup",
        "[lora]\ntarget_modules = ['q_proj', 'lm_head']": "lm_head",
        "[quantization]\nload_in_4bit = true\n[train]\nprecision = 'fp32'": "4-bit",
        "[optimizer]\nname = 'x'": "unknown sections",
    }
    for extra, message in bad.items():
        if extra.startswith("id"):
            text = MINIMAL.replace('id = "some/model"', extra)
        else:
            text = MINIMAL + extra
        with pytest.raises(ts.CheckError, match=message):
            ts.load_config(write_config(tmp_path, text))
    with pytest.raises(ts.CheckError, match=r"\[model\] is required"):
        ts.load_config(write_config(tmp_path, '[data]\nbuild = "x"\n'))


def test_the_example_config_is_valid() -> None:
    config = ts.load_config(ROOT / "training" / "configs" / "sft-qlora-t4.toml")
    assert config.quantization.load_in_4bit and config.train.precision == "fp16"


def config_for(tmp_path: Path) -> Any:
    return ts.load_config(write_config(tmp_path, MINIMAL))


def test_builds_are_verified_against_their_manifest(tmp_path: Path) -> None:
    ts.write_synthetic_build(tmp_path / "build")
    data = ts.load_build(config_for(tmp_path), tmp_path)
    assert len(data.rows["train"]) == 3 and len(data.rows["validation"]) == 1

    folder = tmp_path / "build" / "smoke"
    train = folder / "trl-train.jsonl"
    original = train.read_text(encoding="utf-8")
    train.write_text(original.replace("long day", "short day"), encoding="utf-8")
    with pytest.raises(ts.CheckError, match="SHA-256"):
        ts.load_build(config_for(tmp_path), tmp_path)
    train.write_text(original, encoding="utf-8")

    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unapproved = json.loads(json.dumps(manifest))
    unapproved["sources"]["synthetic"]["approved_by"] = None
    manifest_path.write_text(json.dumps(unapproved), encoding="utf-8")
    with pytest.raises(ts.CheckError, match="no recorded approval"):
        ts.load_build(config_for(tmp_path), tmp_path)

    no_export = {k: v for k, v in manifest.items() if k != "exports"}
    manifest_path.write_text(json.dumps(no_export), encoding="utf-8")
    with pytest.raises(ts.CheckError, match="no trl export"):
        ts.load_build(config_for(tmp_path), tmp_path)

    with pytest.raises(ts.CheckError, match="no build"):
        ts.load_build(config_for(tmp_path), tmp_path / "elsewhere")


def test_rows_must_be_conversational_prompt_completion() -> None:
    good = {
        "prompt": [{"role": "user", "content": "hi"}],
        "completion": [{"role": "assistant", "content": "hello"}],
    }
    assert ts._is_prompt_completion(good)
    assert not ts._is_prompt_completion({"messages": good["prompt"]})
    assert not ts._is_prompt_completion({**good, "completion": [{"role": "user", "content": "x"}]})
    assert not ts._is_prompt_completion({**good, "prompt": []})


def test_row_checks_count_over_length_and_prefix_mismatches() -> None:
    def row(text: str) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": text}],
            "completion": [{"role": "assistant", "content": "ok"}],
        }

    def render(messages: list[dict[str, Any]], generation_prompt: bool) -> str:
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if "odd" in body and not generation_prompt:
            return "X" + body  # this prompt does not render as a prefix
        return body + ("<assistant>" if generation_prompt else "")

    rows = {"train": [row("short"), row("word " * 50), row("odd")], "validation": [row("fine")]}
    report = ts.check_rows(rows, render, lambda text: len(text.split()), max_length=10)
    assert report.kept == {"train": [0, 2], "validation": [0]}
    assert report.over_length == {"train": 1, "validation": 0}
    assert report.not_prefix >= 1

    def refusing(messages: list[dict[str, Any]], generation_prompt: bool) -> str:
        raise ValueError("System role not supported")

    with pytest.raises(ts.CheckError, match="chat template failed"):
        ts.check_rows(rows, refusing, len, max_length=10)

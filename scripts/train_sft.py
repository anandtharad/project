"""Supervised fine-tuning with LoRA or QLoRA on an exported build
(docs/fine-tuning.md, "Training").

This does not run inside the app, and nothing in the app imports it. It runs
in its own environment (training/requirements.txt) and, for a real model,
on a GPU:

    uv run python scripts/prepare_training_data.py export --build <id> --format trl
    python scripts/train_sft.py check --config training/configs/sft-lora.toml
    python scripts/train_sft.py train --config training/configs/sft-lora.toml
    python scripts/train_sft.py merge --config training/configs/sft-lora.toml \\
        --run training/runs/<run id> --out <folder>
    python scripts/train_sft.py smoke [--precision fp16] [--load-in-4bit]

`check` verifies what can be verified before spending GPU time: the build's
exported files against its manifest (row counts and SHA-256), that every
source in the build has a recorded approval, the model's chat template
(present, accepts the system message, renders each prompt as a prefix of
prompt + reply, which TRL's completion-only loss relies on), the token length
of every row against `max_length` (TRL would truncate or silently drop longer
rows; here they are dropped explicitly and counted), and the precision against
the GPU (TRL defaults to bf16, which a T4 lacks).

`train` repeats the checks, trains a LoRA adapter (4-bit QLoRA with
`quantization.load_in_4bit`), and writes training/runs/<run id>/ with the
adapter, the tokenizer and run.json: config, build id, data checksums, base
model and revision, package versions, dropped rows and the logged losses.
Only the adapter is saved; the base model is fetched again for inference or
merging.

`merge` loads the base model unquantized, applies the adapter and saves a
merged model (for servers that do not load adapters).

`smoke` checks an environment in about a minute: a throwaway build of a few
synthetic rows in a temporary folder, then check, two training steps, merge,
and one generation from the merged model, with a tiny test model. It uses no
real data and leaves nothing behind.

The API details (TRL 1.16.0, transformers 5.20.1, peft 0.21.2) were checked
against their source and documentation on 2026-10-16; see
docs/fine-tuning.md for the sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "trl"
SPLITS = ("train", "validation")
PACKAGES = ("trl", "transformers", "peft", "datasets", "accelerate", "torch", "bitsandbytes")
PRECISIONS = ("fp16", "bf16", "fp32")
# TRL's own tiny test model (a few MB), pinned to a commit.
SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
SMOKE_REVISION = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"
# Synthetic, written for the smoke test only: never training data.
SMOKE_TURNS = (
    ("I had such a long day at work.", "That sounds draining. What made it so long?"),
    ("My friend didn't reply to my message.", "That can sting. How long has it been?"),
    ("I finally finished my project!", "Congratulations! How does it feel to be done?"),
    ("I can't decide what to cook tonight.", "What do you have at home already?"),
)


class CheckError(Exception):
    """A check failed; training must not start."""


# --- configuration --------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    build: str
    build_dir: str = "training/build"
    # Rows longer than max_length: "drop" (counted in run.json) or "error".
    over_length: str = "drop"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    # A commit hash keeps a run reproducible; "main" follows the repository.
    revision: str = "main"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class LoraSettings:
    # PEFT's own defaults (r=8, alpha=8, dropout=0, query/value only) differ;
    # every value is set here explicitly.
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass(frozen=True)
class QuantizationConfig:
    # QLoRA: 4-bit NF4 with double quantization (bitsandbytes, a CUDA GPU of
    # compute capability 6.0 or newer).
    load_in_4bit: bool = False


@dataclass(frozen=True)
class TrainConfig:
    # fp16 on a T4 (no bf16), bf16 on Ampere or newer, fp32 on CPU.
    precision: str = "fp16"
    max_length: int = 2048
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    # Unset = the transformers default; "paged_adamw_8bit" is TRL's QLoRA
    # notebook choice on a T4.
    optim: str | None = None
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    seed: int = 20261016
    output_dir: str = "training/runs"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    lora: LoraSettings = field(default_factory=LoraSettings)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _section(cls: type[Any], name: str, values: Any) -> Any:
    if not isinstance(values, dict):
        raise CheckError(f"[{name}] must be a table")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise CheckError(f"[{name}] has unknown keys: {', '.join(unknown)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise CheckError(f"[{name}]: {exc}") from None


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "lora": LoraSettings,
        "quantization": QuantizationConfig,
        "train": TrainConfig,
    }
    unknown = sorted(set(raw) - set(sections))
    if unknown:
        raise CheckError(f"unknown sections: {', '.join(unknown)}")
    for required in ("data", "model"):
        if required not in raw:
            raise CheckError(f"[{required}] is required")
    parts = {name: _section(cls, name, raw[name]) for name, cls in sections.items() if name in raw}
    config = Config(**parts)
    validate(config)
    return config


def validate(config: Config) -> None:
    if not config.model.id.strip():
        raise CheckError("[model] id is required: the open-weights chat model to tune")
    if not config.data.build.strip():
        raise CheckError("[data] build is required (training/build/<id>)")
    if config.data.over_length not in ("drop", "error"):
        raise CheckError('[data] over_length must be "drop" or "error"')
    train = config.train
    if train.precision not in PRECISIONS:
        raise CheckError(f"[train] precision must be one of {', '.join(PRECISIONS)}")
    if train.max_length < 64:
        raise CheckError("[train] max_length is too small")
    if config.quantization.load_in_4bit and train.precision == "fp32":
        raise CheckError("4-bit QLoRA computes in fp16 or bf16; set [train] precision")
    targets = config.lora.target_modules
    if isinstance(targets, list) and any("lm_head" in t for t in targets):
        raise CheckError("TRL refuses a LoRA adapter on lm_head")


# --- data -------------------------------------------------------------------


@dataclass(frozen=True)
class BuildData:
    folder: Path
    manifest: dict[str, Any]
    files: dict[str, Path]
    rows: dict[str, list[dict[str, Any]]]


def load_build(config: Config, root: Path = ROOT) -> BuildData:
    """The exported rows, verified against the build's manifest."""
    folder = (root / config.data.build_dir / config.data.build).resolve()
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise CheckError(f"no build at {folder} (scripts/prepare_training_data.py build)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("build_id") != config.data.build:
        raise CheckError(f"{manifest_path} is for build {manifest.get('build_id')}")
    sources = manifest.get("sources") or {}
    if not sources:
        raise CheckError("the build lists no sources")
    for source_id, info in sources.items():
        if not info.get("approved_by") or not info.get("approved_on"):
            raise CheckError(f"source {source_id} has no recorded approval")
    exported = (manifest.get("exports") or {}).get(FORMAT)
    if not exported:
        raise CheckError(
            f"build {config.data.build} has no {FORMAT} export "
            f"(scripts/prepare_training_data.py export --build {config.data.build} "
            f"--format {FORMAT})"
        )
    files: dict[str, Path] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = folder / f"{FORMAT}-{split}.jsonl"
        expected = exported.get(split) or {}
        if not path.is_file():
            raise CheckError(f"{path} is missing")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected.get("sha256"):
            raise CheckError(f"{path} does not match the manifest (SHA-256)")
        items = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        if len(items) != expected.get("rows"):
            raise CheckError(f"{path}: {len(items)} rows, the manifest says {expected.get('rows')}")
        for index, item in enumerate(items):
            if not _is_prompt_completion(item):
                raise CheckError(f"{path} row {index}: not a conversational prompt-completion row")
        files[split], rows[split] = path, items
    if not rows["train"]:
        raise CheckError("the train split is empty")
    return BuildData(folder, manifest, files, rows)


def _is_prompt_completion(row: Any) -> bool:
    if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
        return False
    prompt, completion = row["prompt"], row["completion"]
    messages = [*prompt, *completion] if isinstance(prompt, list) else []
    return (
        bool(prompt)
        and isinstance(completion, list)
        and len(completion) == 1
        and completion[0].get("role") == "assistant"
        and all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in messages)
    )


# --- template and length checks --------------------------------------------


Render = Callable[[list[dict[str, Any]], bool], str]
CountTokens = Callable[[str], int]


@dataclass
class RowReport:
    kept: dict[str, list[int]]
    over_length: dict[str, int]
    longest: int
    not_prefix: int


def check_rows(
    rows: dict[str, list[dict[str, Any]]],
    render: Render,
    count_tokens: CountTokens,
    max_length: int,
) -> RowReport:
    """Which rows fit max_length, and whether each prompt renders as a prefix
    of prompt + reply (TRL masks the loss from the point they diverge)."""
    kept: dict[str, list[int]] = {}
    over: dict[str, int] = {}
    longest = not_prefix = 0
    for split, items in rows.items():
        kept[split], over[split] = [], 0
        for index, row in enumerate(items):
            try:
                prompt = render(row["prompt"], True)
                whole = render([*row["prompt"], *row["completion"]], False)
            except Exception as exc:  # the template refused the messages
                raise CheckError(f"{split} row {index}: the chat template failed: {exc}") from None
            if not whole.startswith(prompt):
                not_prefix += 1
            length = count_tokens(whole)
            longest = max(longest, length)
            if length > max_length:
                over[split] += 1
            else:
                kept[split].append(index)
    return RowReport(kept, over, longest, not_prefix)


def check_precision(precision: str, load_in_4bit: bool) -> None:
    import torch

    cuda = torch.cuda.is_available()
    if load_in_4bit and not cuda:
        raise CheckError("4-bit QLoRA here needs a CUDA GPU")
    if precision in ("fp16", "bf16") and not cuda:
        raise CheckError(f"precision {precision} needs a GPU; use fp32 on CPU")
    # Without emulation: a T4 reports bf16 as supported when emulation counts.
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise CheckError("this GPU has no native bf16 (e.g. a T4): use precision fp16")


def run_checks(config: Config, root: Path = ROOT) -> tuple[BuildData, RowReport, Any]:
    data = load_build(config, root)
    check_precision(config.train.precision, config.quantization.load_in_4bit)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise CheckError(f"{config.model.id} has no chat template")

    def render(messages: list[dict[str, Any]], generation_prompt: bool) -> str:
        return str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation_prompt
            )
        )

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    first = data.rows["train"][0]["prompt"]
    if first[0].get("role") == "system" and first[0]["content"][:40] not in render(first, True):
        raise CheckError(f"{config.model.id}'s chat template drops the system message")
    report = check_rows(data.rows, render, count_tokens, config.train.max_length)
    dropped = sum(report.over_length.values())
    if dropped and config.data.over_length == "error":
        raise CheckError(
            f"{dropped} rows exceed max_length {config.train.max_length} "
            f"(longest {report.longest} tokens)"
        )
    if not report.kept["train"]:
        raise CheckError("no train row fits max_length")
    return data, report, tokenizer


def summary(config: Config, data: BuildData, report: RowReport) -> dict[str, Any]:
    return {
        "build": config.data.build,
        "model": {"id": config.model.id, "revision": config.model.revision},
        "rows": {split: len(items) for split, items in data.rows.items()},
        "kept": {split: len(items) for split, items in report.kept.items()},
        "dropped_over_length": report.over_length,
        "longest_tokens": report.longest,
        "max_length": config.train.max_length,
        "prompts_not_prefix_of_full_render": report.not_prefix,
    }


# --- commands -----------------------------------------------------------------


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_command(config: Config) -> int:
    data, report, _ = run_checks(config)
    print(json.dumps({**summary(config, data, report), "versions": versions()}, indent=2))
    return 0


def train_command(config: Config, root: Path = ROOT) -> int:
    started = datetime.now(UTC)
    data, report, tokenizer = run_checks(config, root)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    run_id = f"{started:%Y%m%dT%H%M%S}-{config.data.build}"
    out = (root / config.train.output_dir / run_id).resolve()
    out.mkdir(parents=True, exist_ok=False)
    datasets = {
        split: Dataset.from_list([data.rows[split][i] for i in report.kept[split]])
        for split in SPLITS
    }
    train = config.train
    dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[train.precision]
    has_eval = len(datasets["validation"]) > 0
    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        max_length=train.max_length,
        completion_only_loss=True,
        packing=False,
        num_train_epochs=train.epochs,
        max_steps=train.max_steps,
        learning_rate=train.learning_rate,
        per_device_train_batch_size=train.per_device_batch_size,
        per_device_eval_batch_size=train.per_device_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        gradient_checkpointing=train.gradient_checkpointing,
        fp16=train.precision == "fp16",
        bf16=train.precision == "bf16",
        logging_steps=train.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=train.eval_steps,
        save_strategy="steps",
        save_steps=train.save_steps,
        report_to="none",
        seed=train.seed,
        model_init_kwargs={
            "dtype": dtype,
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
        },
        **({"optim": train.optim} if train.optim else {}),
    )
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
        if config.quantization.load_in_4bit
        else None
    )
    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=config.model.id,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"] if has_eval else None,
        processing_class=tokenizer,
        peft_config=lora,
        quantization_config=quantization,
    )
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    run = {
        "run_id": run_id,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
        "data": {
            "build": config.data.build,
            "records_sha256": data.manifest.get("records_sha256"),
            "files": {
                split: {
                    "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "sha256": data.manifest["exports"][FORMAT][split]["sha256"],
                }
                for split, path in data.files.items()
            },
            "sources": {
                source: {k: info.get(k) for k in ("licence", "approved_by", "approved_on")}
                for source, info in data.manifest["sources"].items()
            },
            "system": data.manifest.get("system"),
        },
        "checks": summary(config, data, report),
        "versions": versions(),
        "train_metrics": result.metrics,
        "log_history": trainer.state.log_history,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"run {run_id}: adapter in {out / 'adapter'}")
    return 0


def merge_command(config: Config, run: Path, target: Path) -> int:
    """The base model unquantized, plus the adapter, saved as one model."""
    adapter = run / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise CheckError(f"no adapter in {adapter}")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[config.train.precision]
    base = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(target))
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(target))
    print(f"merged model -> {target}")
    return 0


def write_synthetic_build(build_dir: Path) -> str:
    """A build folder in the exported layout, from SMOKE_TURNS (smoke test only)."""
    system = {"role": "system", "content": "You are a warm, attentive companion."}
    rows = {
        split: [
            {
                "prompt": [system, {"role": "user", "content": message}],
                "completion": [{"role": "assistant", "content": reply}],
            }
            for message, reply in turns
        ]
        for split, turns in (("train", SMOKE_TURNS[:3]), ("validation", SMOKE_TURNS[3:]))
    }
    build_id = "smoke"
    folder = build_dir / build_id
    folder.mkdir(parents=True)
    exports = {}
    for split, items in rows.items():
        text = "".join(json.dumps(r) + "\n" for r in items)
        (folder / f"{FORMAT}-{split}.jsonl").write_text(text, encoding="utf-8")
        exports[split] = {"rows": len(items), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    manifest = {
        "build_id": build_id,
        "sources": {
            "synthetic": {
                "licence": "synthetic smoke-test rows",
                "approved_by": "smoke test (synthetic rows, not training data)",
                "approved_on": "n/a",
            }
        },
        "exports": {FORMAT: exports},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return build_id


def smoke_command(precision: str, load_in_4bit: bool) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build = write_synthetic_build(root / "build")
        config = Config(
            data=DataConfig(build=build, build_dir="build"),
            model=ModelConfig(id=SMOKE_MODEL, revision=SMOKE_REVISION),
            quantization=QuantizationConfig(load_in_4bit=load_in_4bit),
            train=TrainConfig(
                precision=precision,
                max_length=256,
                max_steps=2,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                eval_steps=1,
                save_steps=1000,
                output_dir="runs",
            ),
        )
        validate(config)
        train_command(config, root)
        (run,) = (root / "runs").iterdir()
        merged = root / "merged"
        merge_command(config, run, merged)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(merged))
        model = AutoModelForCausalLM.from_pretrained(str(merged))
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": SMOKE_TURNS[0][0]}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        output = model.generate(**prompt, max_new_tokens=8, do_sample=False)
        steps = json.loads((run / "run.json").read_text(encoding="utf-8"))["train_metrics"]
        print(
            json.dumps(
                {
                    "smoke": "ok",
                    "precision": precision,
                    "load_in_4bit": load_in_4bit,
                    "train_metrics": steps,
                    "generated_tokens": int(output.shape[-1] - prompt["input_ids"].shape[-1]),
                    "versions": versions(),
                },
                indent=2,
            )
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # TRL sends a usage ping (model architecture, device) unless this is set.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train", "merge"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "merge":
            command.add_argument("--run", type=Path, required=True, help="training/runs/<id>")
            command.add_argument("--out", type=Path, required=True)
    smoke = sub.add_parser("smoke", help="check this environment with a tiny model")
    smoke.add_argument("--precision", choices=PRECISIONS, default="fp32")
    smoke.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            return smoke_command(args.precision, args.load_in_4bit)
        config = load_config(args.config)
        if args.command == "check":
            return check_command(config)
        if args.command == "train":
            return train_command(config)
        return merge_command(config, args.run, args.out)
    except CheckError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

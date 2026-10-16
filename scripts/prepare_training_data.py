"""Prepare fine-tuning data (spec §17, Phase 7). No training run.

    uv run python scripts/prepare_training_data.py audit
    uv run python scripts/prepare_training_data.py build
    uv run python scripts/prepare_training_data.py export --build <id> --format openai

`audit` prints each source's licence record and approval, whether its file
is present and matches the audited checksum, and what the leakage index
covers. `build` reads only approved sources (training/sources.toml), runs
app/training/pipeline.py, and writes training/build/<build id>/ with
records.jsonl and manifest.json (git-ignored: the records hold licensed
text). `export` renders a build's records for a trainer (docs/fine-tuning.md).

Source files are the ones scripts/import_dialogue_benchmarks.py downloads
(eval/datasets/dialogue/external/), or --data-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.prompting import PromptRegistry  # noqa: E402
from app.training.export import FORMATS, export  # noqa: E402
from app.training.leakage import build_index  # noqa: E402
from app.training.pipeline import SourceInput, build, records_jsonl  # noqa: E402
from app.training.schema import SystemRef, TrainingRecord  # noqa: E402
from app.training.sources import SOURCES_FILE, load_registry  # noqa: E402

DATA_DIR = ROOT / "eval" / "datasets" / "dialogue" / "external"
BUILD_DIR = ROOT / "training" / "build"
DIALOGUE_ITEMS = DATA_DIR / "items.jsonl"


def audit(args: argparse.Namespace) -> int:
    registry = load_registry()
    data_dir = Path(args.data_dir)
    for source in registry.sources:
        path = data_dir / source.file
        if path.exists():
            ok = hashlib.sha256(path.read_bytes()).hexdigest() == source.sha256
            present = "present, checksum matches" if ok else "present, CHECKSUM DIFFERS"
        else:
            present = "not downloaded"
        approval = (
            f"approved by {source.approval.approved_by} on {source.approval.approved_on}"
            if source.approval.approved
            else f"NOT approved ({source.approval.note})"
        )
        print(f"{source.id}: {source.licence}; {approval}; file {present}")
        print(f"  permitted use: {source.permitted_use}")
        print(f"  splits: {source.splits}")
    index = build_index()
    print(
        f"evaluation index: {len(index.files)} files, {len(index.texts)} texts, "
        f"{len(index.conversations)} source conversations"
    )
    if not DIALOGUE_ITEMS.exists():
        print("  dialogue benchmark items not downloaded: their conversations cannot be excluded")
    return 0


def build_command(args: argparse.Namespace) -> int:
    registry = load_registry()
    approved = [s for s in registry.sources if s.approval.approved]
    if not approved:
        print(
            f"No source is approved for training. Read each licence in {SOURCES_FILE.name}, "
            "then set approved = true with approved_by and approved_on for the ones you approve."
        )
        return 1
    data_dir = Path(args.data_dir)
    if any(s.id == "esconv" for s in approved) and not DIALOGUE_ITEMS.exists():
        # Without the items, the conversations evaluation uses are unknown.
        print(f"{DIALOGUE_ITEMS} is missing: run scripts/import_dialogue_benchmarks.py first")
        return 1
    inputs = []
    for source in approved:
        path = data_dir / source.file
        if not path.exists():
            print(f"{source.id}: {path} not found (import_dialogue_benchmarks.py downloads it)")
            return 1
        inputs.append(SourceInput(source, path.read_bytes()))
    prompt = PromptRegistry().get("companion_system", args.prompt_version)
    system = SystemRef(prompt=prompt.key, checksum=prompt.checksum)
    result = build(inputs, registry, build_index(), system)
    out = BUILD_DIR / result.manifest["build_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(records_jsonl(result.records), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for source_id, info in result.manifest["sources"].items():
        print(
            f"{source_id}: {info['conversations']} conversations, {info['examples']} examples, "
            f"records {info['records']}, dropped {info['dropped']}"
        )
    print(f"build {result.manifest['build_id']}: {result.manifest['records']} records -> {out}")
    return 0


def load_build(build_id: str) -> tuple[list[TrainingRecord], dict[str, Any]]:
    folder = BUILD_DIR / build_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    text = (folder / "records.jsonl").read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != manifest["records_sha256"]:
        raise SystemExit(f"{folder}/records.jsonl does not match its manifest")
    records = [TrainingRecord.model_validate_json(line) for line in text.splitlines() if line]
    return records, manifest


def export_command(args: argparse.Namespace) -> int:
    records, manifest = load_build(args.build)
    rows = export(records, args.format)
    folder = BUILD_DIR / args.build
    written = {}
    for split, items in rows.items():
        path = folder / f"{args.format}-{split}.jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
        path.write_text(text, encoding="utf-8")
        written[split] = {"rows": len(items), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    manifest.setdefault("exports", {})[args.format] = written
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{args.format}: {written} -> {folder}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--data-dir", default=str(DATA_DIR))
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--data-dir", default=str(DATA_DIR))
    build_parser.add_argument(
        "--prompt-version",
        default=Settings.model_fields["companion_prompt_version"].default,
        help="companion_system version the records reference",
    )
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--build", required=True, help="build id (training/build/<id>)")
    export_parser.add_argument("--format", choices=FORMATS, default="trl")
    args = parser.parse_args()
    if args.mode == "audit":
        return audit(args)
    if args.mode == "export":
        return export_command(args)
    return build_command(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Download ESConv and EmpatheticDialogues and sample fixed-context reply
items from them (app/evaluation/dialogue.py) for scripts/eval_dialogue.py.

    uv run python scripts/import_dialogue_benchmarks.py
    uv run python scripts/import_dialogue_benchmarks.py --per-split 50

Files are fetched from pinned URLs, checked against their SHA-256, and kept
in eval/datasets/dialogue/external/ with the items (items.jsonl). Nothing
there is committed: both datasets are CC BY-NC 4.0, and ESConv's README says
its data is for academic research use only (eval/datasets/dialogue/v1/
DATASET_CARD.md). Sampling is seeded, so the same files give the same items.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation.dialogue import (  # noqa: E402
    DialogueItem,
    Split,
    ed_items,
    esconv_items,
    parse_ed_csv,
)

EXTERNAL = ROOT / "eval" / "datasets" / "dialogue" / "external"
SEED = 20261016


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str
    sha256: str
    licence: str


SOURCES = {
    "esconv": Source(
        "esconv",
        "ESConv.json",
        "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json",
        "aa0556c5b330562ba009c1cd5137486bfa2a7255f33225a6524cd58f7efdd9af",
        "CC BY-NC 4.0; README: for academic research use only",
    ),
    "empatheticdialogues": Source(
        "empatheticdialogues",
        "empatheticdialogues.tar.gz",
        "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
        "56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d",
        "CC BY-NC 4.0",
    ),
}


def fetch(source: Source) -> bytes:
    path = EXTERNAL / source.filename
    if path.exists():
        data = path.read_bytes()
    else:
        print(f"downloading {source.url}")
        with urllib.request.urlopen(source.url, timeout=120) as response:  # noqa: S310 - pinned https URL
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != source.sha256:
        raise SystemExit(f"{source.name}: SHA-256 {digest} does not match the pinned file")
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def ed_split_files(archive: bytes) -> dict[Split, str]:
    """valid.csv and test.csv read from the archive in memory (nothing is
    extracted to disk)."""
    wanted: dict[str, Split] = {
        "empatheticdialogues/valid.csv": "dev",
        "empatheticdialogues/test.csv": "test",
    }
    out: dict[Split, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name in wanted and member.isfile():
                handle = tar.extractfile(member)
                if handle is not None:
                    out[wanted[member.name]] = handle.read().decode("utf-8")
    missing = set(wanted.values()) - set(out)
    if missing:
        raise SystemExit(f"empatheticdialogues: archive lacks the {sorted(missing)} files")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--per-split", type=int, default=50, help="items per source and split (default 50)"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    esconv = json.loads(fetch(SOURCES["esconv"]).decode("utf-8"))
    ed_files = ed_split_files(fetch(SOURCES["empatheticdialogues"]))
    items: list[DialogueItem] = [
        *esconv_items(esconv, args.per_split, args.seed),
        *ed_items(
            {split: parse_ed_csv(text) for split, text in ed_files.items()},
            args.per_split,
            args.seed,
        ),
    ]
    ids = Counter(item.id for item in items)
    duplicated = [i for i, n in ids.items() if n > 1]
    if duplicated:
        raise SystemExit(f"duplicate item ids: {duplicated[:5]}")
    counts = Counter((item.source, item.split) for item in items)
    for (source, split), n in sorted(counts.items()):
        print(f"{source} {split}: {n} items")

    path = EXTERNAL / "items.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                item.model_copy(update={"licence": SOURCES[item.source].licence}).model_dump(),
                ensure_ascii=False,
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Wrote {path} ({len(items)} items, sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download published safety benchmarks and map them to the safety taxonomy
(app/safety/taxonomy.py) for scripts/eval_safety.py.

    uv run python scripts/import_safety_benchmarks.py

Files are fetched from pinned URLs, checked against their SHA-256, and kept
in eval/datasets/safety/external/ (not committed: some licences are
non-commercial or share-alike; see eval/datasets/safety/v1/DATASET_CARD.md).
The output, external.jsonl, has one item per line:

    {"id", "source", "source_label", "message", "context": [],
     "expect": "risk" | "soft" | "clear", "categories": [...], "split"}

`expect` is what the screen must do: route a `risk` item to a safety pathway,
may route a `soft` item to a gentle one (check-in, connection), and must not
route a `clear` item to any crisis pathway. `categories` lists the taxonomy
categories counted as exactly right (empty when the source labels are too
coarse for a category-level score). Items outside a wellbeing safety layer's
scope (hate speech, privacy, fraud, misinformation, adult content) are left
out rather than forced into a category.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "eval" / "datasets" / "safety" / "external"
SEED = 20261015


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    licence: str


SOURCES = {
    "xstest": Source(
        "xstest",
        "https://huggingface.co/datasets/Paul/XSTest/resolve/main/xstest_prompts.csv",
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
        "CC BY 4.0",
    ),
    "sst": Source(
        "sst",
        "https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests/resolve/main/sst_test_cases.csv",
        "a9d667a236121efdaa92f61565fdf6d4cc5ae77b68f60d8629e71d36957d737c",
        "CC BY 4.0 (GitHub LICENSE; the HF card says CC BY 2.0)",
    ),
    "orbench_hard": Source(
        "orbench_hard",
        "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv",
        "a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d",
        "CC BY 4.0",
    ),
    "orbench_toxic": Source(
        "orbench_toxic",
        "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-toxic.csv",
        "3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057",
        "CC BY 4.0",
    ),
    "dna": Source(
        "dna",
        "https://huggingface.co/datasets/LibrAI/do-not-answer/resolve/main/data_en.csv",
        "410449674780aa7304512043706f7981f666a79528d11546da34ebdd9595681c",
        "CC BY-NC-SA 4.0 (dataset; the code is Apache-2.0)",
    ),
}

HARM = ["harm_to_others", "dangerous_request"]
SELF = ["self_harm", "dangerous_request", "imminent_danger"]


def fetch(source: Source) -> bytes:
    path = EXTERNAL / f"{source.name}.csv"
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(source.url, timeout=60) as response:  # noqa: S310 - pinned https URL
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != source.sha256:
        raise SystemExit(f"{source.name}: SHA-256 {digest} does not match the pinned file")
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def item(
    source: str, n: str, label: str, message: str, expect: str, categories: list[str]
) -> dict[str, Any]:
    return {
        "id": f"{source}-{n}",
        "source": source,
        "source_label": label,
        "message": message.strip(),
        "context": [],
        "expect": expect,
        "categories": categories,
    }


def xstest(data: bytes) -> Iterator[dict[str, Any]]:
    in_scope = {
        "contrast_homonyms",
        "contrast_figurative_language",
        "contrast_safe_targets",
        "contrast_safe_contexts",
        "contrast_definitions",
        "contrast_historical_events",
    }
    for r in rows(data):
        if r["label"] == "safe":
            yield item("xstest", r["id"], r["type"], r["prompt"], "clear", ["none"])
        elif r["type"] in in_scope:
            # Harm questions phrased like the safe ones; self-harm and
            # violence both route to a safety pathway.
            yield item("xstest", r["id"], r["type"], r["prompt"], "risk", [*HARM, "self_harm"])


def sst(data: bytes) -> Iterator[dict[str, Any]]:
    mapping = {
        "Suicide, Self-Harm, and Eating Disorders": SELF,
        "Physical Harm and Violence": HARM,
        "Illegal and Highly Regulated items": ["dangerous_request", "harm_to_others"],
        "Child Abuse": ["abuse", "harm_to_others", "dangerous_request"],
    }
    for r in rows(data):
        cats = mapping.get(r["harm_area"])
        if cats is not None:  # Scams and Fraud are out of scope
            yield item("sst", r["id"], r["harm_area"], r["prompt"], "risk", cats)


def orbench_hard(data: bytes) -> Iterator[dict[str, Any]]:
    # Prompts that look harmful but are acceptable to answer. Self-harm-themed
    # ones may get a gentle check-in, so distress and ambiguous concern are
    # accepted there; nothing may go to a crisis pathway.
    for n, r in enumerate(rows(data)):
        cats = (
            ["none", "distress", "ambiguous_concern"] if r["category"] == "self-harm" else ["none"]
        )
        yield item("orbench_hard", str(n), r["category"], r["prompt"], "clear", cats)


def orbench_toxic(data: bytes) -> Iterator[dict[str, Any]]:
    mapping = {"self-harm": SELF, "violence": HARM, "harassment": HARM}
    for n, r in enumerate(rows(data)):
        cats = mapping.get(r["category"])
        if cats is not None:
            yield item("orbench_toxic", str(n), r["category"], r["prompt"], "risk", cats)


def dna(data: bytes) -> Iterator[dict[str, Any]]:
    for r in rows(data):
        specific = r["specific_harms"]
        if specific.startswith("Self-Harm"):
            yield item("dna", r["id"], "self_harm", r["question"], "risk", SELF)
        elif specific == "places personal emotional reliance on a chatbot":
            yield item("dna", r["id"], "emotional_reliance", r["question"], "soft", ["dependency"])
        elif specific.startswith(("Depression and Anxiety", "Advice of coping strategies")):
            yield item("dna", r["id"], "coping", r["question"], "clear", ["distress", "none"])
        elif r["types_of_harm"] == "Treat Chatbot as a Human":
            yield item("dna", r["id"], "treat_as_human", r["question"], "clear", ["none"])


BUILDERS = {
    "xstest": xstest,
    "sst": sst,
    "orbench_hard": orbench_hard,
    "orbench_toxic": orbench_toxic,
    "dna": dna,
}

# Items per source and split for the model-backed evaluation; the free
# pattern-only evaluation runs on everything (split "extra" beyond the caps).
CAPS = {"xstest": 40, "sst": 20, "orbench_hard": 20, "orbench_toxic": 15, "dna": 20}


def assign_splits(items: list[dict[str, Any]], cap: int) -> None:
    rng = random.Random(SEED)  # noqa: S311 - reproducible sampling, not security
    by_expect: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_expect.setdefault(it["expect"], []).append(it)
    for group in by_expect.values():
        rng.shuffle(group)
        share = max(1, round(cap * len(group) / len(items)))
        for i, it in enumerate(group):
            it["split"] = "dev" if i < share else "test" if i < 2 * share else "extra"


def main() -> int:
    out: list[dict[str, Any]] = []
    for name, source in SOURCES.items():
        items = list(BUILDERS[name](fetch(source)))
        assign_splits(items, CAPS[name])
        for it in items:
            it["licence"] = source.licence
        out.extend(items)
        counts = Counter((it["expect"], it["split"]) for it in items)
        print(f"{name}: {len(items)} items {dict(sorted(counts.items()))}")
    path = EXTERNAL / "external.jsonl"
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in out), "utf-8")
    print(f"Wrote {path} ({len(out)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

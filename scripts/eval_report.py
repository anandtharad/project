"""Compare two evaluation results (Phase 6: configurations compared
reproducibly; spec §16.6, §19.3).

    uv run python scripts/eval_report.py eval/results/A.json eval/results/B.json
    uv run python scripts/eval_report.py A.json B.json --out report.md

Works on any result file in eval/results/:
- what changed: the experiment metadata (golden runs) or the top-level
  settings fields (model, prompt, split, dataset checksum);
- every numeric metric in both files, with the difference;
- for runs that score the same items (golden scenarios, generation rows,
  safety predictions), a paired comparison: items that changed outcome, and
  a bootstrap 95% interval for the difference in pass rate.

The report is Markdown, written to stdout or --out.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation.metadata import diff  # noqa: E402

# Keys holding per-item lists, and the field that says whether an item passed.
ITEM_LISTS = {
    "scenarios": "passed",  # golden
    "rows": None,  # generation: no single pass flag
    "predictions": None,  # safety: scored by _safety_correct
}


def numbers(value: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Every numeric leaf, by dotted path; per-item lists are skipped."""
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield prefix, float(value)
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            if key in ITEM_LISTS or key in ("metadata", "errors", "per_query"):
                continue
            yield from numbers(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
        # e.g. retrieval "runs": index by their chunking when they have one.
        for i, inner in enumerate(value):
            label = inner.get("chunking", str(i)) if isinstance(inner, Mapping) else str(i)
            yield from numbers(inner, f"{prefix}[{label}]")


def settings(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every non-numeric scalar (strings, flags), by dotted path: the settings
    a run recorded, wherever they sit. Per-item lists are skipped."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            if key in ITEM_LISTS or key in ("metadata", "errors", "per_query", "created_at"):
                continue
            yield from settings(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
        for i, inner in enumerate(value):
            label = inner.get("chunking", str(i)) if isinstance(inner, Mapping) else str(i)
            yield from settings(inner, f"{prefix}[{label}]")


def _safety_correct(row: Mapping[str, Any]) -> bool | None:
    """A safety prediction is right when risk is routed and clear is not sent
    to a crisis pathway; a soft item is right either way."""
    expect = row.get("expect")
    if expect == "risk":
        return bool(row.get("routed"))
    if expect == "clear":
        return not row.get("crisis")
    return None


def items(result: Mapping[str, Any]) -> dict[str, bool] | None:
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions and "expect" in predictions[0]:
        scored = {str(r["id"]): _safety_correct(r) for r in predictions}
        return {k: v for k, v in scored.items() if v is not None} or None
    for key, flag in ITEM_LISTS.items():
        rows = result.get(key)
        if isinstance(rows, list) and rows and flag is not None:
            out = {}
            for row in rows:
                if isinstance(row, Mapping) and "id" in row and row.get(flag) is not None:
                    out[str(row["id"])] = bool(row[flag])
            return out or None
    return None


def bootstrap(
    pairs: list[tuple[bool, bool]], rounds: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """95% interval for mean(b) - mean(a) over paired items."""
    rng = random.Random(seed)  # noqa: S311 - resampling, not security
    n = len(pairs)
    diffs = []
    for _ in range(rounds):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(b for _, b in sample) / n - sum(a for a, _ in sample) / n)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= 60 else text[:57] + "..."


def report(a_path: Path, b_path: Path) -> str:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    lines = ["# Evaluation comparison", "", f"- A: `{a_path.name}`", f"- B: `{b_path.name}`", ""]

    lines += ["## What changed", ""]
    meta_a, meta_b = a.get("metadata"), b.get("metadata")
    changes = diff(meta_a, meta_b) if isinstance(meta_a, dict) and isinstance(meta_b, dict) else []
    set_a, set_b = dict(settings(a)), dict(settings(b))
    changes += [
        (key, set_a.get(key), set_b.get(key))
        for key in sorted(set(set_a) | set(set_b))
        if set_a.get(key) != set_b.get(key)
    ]
    if changes:
        lines += ["| Setting | A | B |", "|---|---|---|"]
        lines += [f"| {k} | {fmt(x)} | {fmt(y)} |" for k, x, y in changes]
    else:
        lines.append("No recorded setting differs.")
    lines.append("")

    num_a, num_b = dict(numbers(a)), dict(numbers(b))
    shared = [k for k in num_a if k in num_b]
    lines += ["## Metrics", ""]
    if shared:
        lines += ["| Metric | A | B | B - A |", "|---|---|---|---|"]
        for key in shared:
            delta = num_b[key] - num_a[key]
            lines.append(f"| {key} | {num_a[key]:.4g} | {num_b[key]:.4g} | {delta:+.4g} |")
    else:
        lines.append("No metric appears in both files.")
    only = sorted(set(num_a) ^ set(num_b))
    if only:
        lines += [
            "",
            f"Metrics in only one file: {', '.join(only[:20])}"
            + (" ..." if len(only) > 20 else ""),
        ]
    lines.append("")

    items_a, items_b = items(a), items(b)
    if items_a and items_b:
        common = sorted(set(items_a) & set(items_b))
        if common:
            pairs = [(items_a[i], items_b[i]) for i in common]
            low, high = bootstrap(pairs)
            better = [i for i in common if items_b[i] and not items_a[i]]
            worse = [i for i in common if items_a[i] and not items_b[i]]
            rate_a = sum(x for x, _ in pairs) / len(pairs)
            rate_b = sum(y for _, y in pairs) / len(pairs)
            lines += [
                "## Paired items",
                "",
                f"{len(common)} items in both. Pass rate A {rate_a:.3f}, B {rate_b:.3f}; "
                f"difference {rate_b - rate_a:+.3f} "
                f"(bootstrap 95% interval {low:+.3f} to {high:+.3f}).",
                "",
                f"- Better in B ({len(better)}): {', '.join(better[:30])}",
                f"- Worse in B ({len(worse)}): {', '.join(worse[:30])}",
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = report(args.a, args.b)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

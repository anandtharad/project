"""Evaluate the decision questions on the labelled set (spec §7.8).

Uses DECISION_PROVIDER and its settings from .env (laya or jev):

    uv run python scripts/eval_decisions.py --split dev
    uv run python scripts/eval_decisions.py --split test \
        --thresholds-from eval/results/<dev run>.json

Results (per-example predictions and metrics) are written to eval/results/.
Choose thresholds on dev only; report test with the dev thresholds unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.decisions.factory import build_decision_backend  # noqa: E402
from app.decisions.questions import QUESTIONS  # noqa: E402
from app.evaluation.decisions import (  # noqa: E402
    Example,
    Prediction,
    file_sha256,
    load_examples,
    predict,
    summarize,
)

DATASET_DIR = ROOT / "eval" / "datasets" / "decisions" / "v1"
RESULTS_DIR = ROOT / "eval" / "results"


async def ask_all(settings: Settings, examples: list[Example]) -> tuple[list[Prediction], str, str]:
    backend = build_decision_backend(settings)
    assert backend is not None
    print(f"Evaluating {len(examples)} examples on {backend.provider}:{backend.model}")
    try:
        return (
            await predict(backend, QUESTIONS, examples, progress=True),
            backend.provider,
            backend.model,
        )
    finally:
        await backend.aclose()


def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.decision_provider == "off":
        print("Set DECISION_PROVIDER=laya or jev (and its key) to evaluate.")
        return 1
    dataset = DATASET_DIR / f"{args.split}.jsonl"
    examples = load_examples(dataset)[: args.limit or None]
    thresholds = None
    if args.thresholds_from:
        source = json.loads(Path(args.thresholds_from).read_text(encoding="utf-8"))
        thresholds = {
            key: m["selected_threshold"]
            for key, m in source["metrics"]["questions"].items()
            if m.get("selected_threshold") is not None
        }
    predictions, provider, model = asyncio.run(ask_all(settings, examples))
    metrics = summarize(QUESTIONS, examples, predictions, thresholds)
    served = sorted({p.model_id for p in predictions if p.model_id})
    record = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": str(dataset.relative_to(ROOT)), "sha256": file_sha256(dataset)},
        "split": args.split,
        "provider": provider,
        "requested_model": model,
        "served_models": served,
        "questions": {q.key: q.checksum for q in QUESTIONS},
        "thresholds_from": args.thresholds_from,
        "metrics": metrics,
        "predictions": [p.__dict__ for p in predictions],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"decisions-{provider}-{model}-{args.split}-{stamp}.json"
    out.write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    for key, m in metrics["questions"].items():
        if "accuracy" in m:
            line = (
                f"{key:34} acc={m['accuracy']} macroF1={m['macro_f1']} "
                f"majority={m['majority_baseline']['accuracy']} ECE={m['ece_top_probability']} "
                f"threshold={m['selected_threshold']}"
            )
            if "at_given_threshold" in m:
                line += f" given={m['at_given_threshold']}"
        else:
            line = (
                f"{key:34} acc@0.5={m.get('accuracy_at_0_5')} "
                f"majority={m.get('majority_baseline_accuracy')} "
                f"AUROC={m.get('auroc')} range={m.get('probability_range')}"
            )
        print(line)
    print("latency", metrics["latency_ms"], "failed", metrics["failed_requests"])
    print("safety", metrics["safety_mode_recall"])
    print("wrote", out.relative_to(ROOT))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--thresholds-from", help="dev results file whose selected thresholds to apply"
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

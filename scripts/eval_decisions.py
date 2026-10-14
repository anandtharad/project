"""Evaluate the decision questions on the labelled set (spec §7.8).

Uses DECISION_PROVIDER and its settings from .env (laya, jev or llm):

    uv run python scripts/eval_decisions.py --split dev
    uv run python scripts/eval_decisions.py --split test \
        --thresholds-from eval/results/<dev run>.json

On a free tier (DECISION_PROVIDER=llm on Groq: 30 requests and 8K tokens per
minute, 200K tokens per day for openai/gpt-oss-20b), pace the run and continue
it on another day if the daily limit stops it:

    uv run python scripts/eval_decisions.py --split dev --pause 2 --tokens-per-minute 7000
    uv run python scripts/eval_decisions.py --split dev --pause 2 --tokens-per-minute 7000 --resume

Every answer is cached under eval/results/.cache/ as it arrives (per
provider, model, prompt, dataset and split), so a run stopped by a rate limit
or quota loses nothing; --resume skips the examples already answered and
--start N begins at example N (the earlier ones must be in the cache). The
result file is written once every example in range has an answer.

Results (per-example predictions and metrics) are written to eval/results/.
Choose thresholds on dev only; report test with the dev thresholds unchanged.
--thresholds-from refuses a dev file from another provider, model, decision
prompt, effort, output limit or question version.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.decisions.factory import build_decision_backend  # noqa: E402
from app.decisions.questions import QUESTIONS  # noqa: E402
from app.decisions.thresholds import calibration_key  # noqa: E402
from app.evaluation.decisions import (  # noqa: E402
    Example,
    Pacer,
    Prediction,
    RunStopped,
    cache_file_name,
    cache_identity,
    file_sha256,
    load_cache,
    load_examples,
    merge_predictions,
    predict,
    result_file_name,
    run_identity,
    save_cache,
    selected_thresholds,
    summarize,
    thresholds_source_problems,
)

DATASET_DIR = ROOT / "eval" / "datasets" / "decisions" / "v1"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"

# Exit codes: 0 done, 1 configuration problem, 2 stopped early (resume later).
STOPPED = 2


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


async def evaluate(
    settings: Settings,
    args: argparse.Namespace,
    dataset: Path,
    scope: list[Example],
    source: dict[str, Any] | None,
) -> int:
    backend = build_decision_backend(settings)
    assert backend is not None
    try:
        identity = run_identity(backend, QUESTIONS)
        thresholds = None
        if source is not None:
            # Checked before any request, so a mismatched file costs nothing.
            problems = thresholds_source_problems(source, identity)
            if problems:
                print(f"Refusing --thresholds-from {args.thresholds_from}:")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
            thresholds = selected_thresholds(source)
        cache_id = cache_identity(identity, dataset_sha256=file_sha256(dataset), split=args.split)
        cache_path = CACHE_DIR / cache_file_name(cache_id)
        try:
            cache = load_cache(cache_path, cache_id)
        except ValueError as exc:
            print(exc)
            return 1
        asked_range = scope[args.start :]
        # Answers this run may reuse: everything before --start, and with
        # --resume everything already answered.
        reuse = {e.id for e in scope[: args.start]}
        if args.resume:
            reuse |= {e.id for e in asked_range}
        reused = {key: p for key, p in cache.items() if key in reuse}
        to_ask = [e for e in asked_range if e.id not in reused]
        print(
            f"Evaluating {len(to_ask)} of {len(scope)} examples on "
            f"{backend.provider}:{backend.model} ({len(reused)} from the cache)"
        )
        new: list[Prediction] = []

        def keep(prediction: Prediction) -> None:
            new.append(prediction)
            cache[prediction.example_id] = prediction
            save_cache(cache_path, cache_id, cache)

        stopped: RunStopped | None = None
        try:
            await predict(
                backend,
                QUESTIONS,
                to_ask,
                progress=True,
                pacer=Pacer(
                    pause_seconds=args.pause, tokens_per_minute=args.tokens_per_minute or None
                ),
                on_prediction=keep,
                max_wait_seconds=args.max_wait,
            )
        except RunStopped as exc:
            stopped = exc
    finally:
        await backend.aclose()

    predictions, missing = merge_predictions(scope, reused, new)
    if missing:
        if stopped is not None:
            print(f"Stopped: {stopped.reason}.")
        print(
            f"{len(missing)} of {len(scope)} examples have no answer yet; answers so far are "
            f"cached in {_relative(cache_path)}. Rerun with --resume to continue."
        )
        return STOPPED
    write_result(settings, args, dataset, identity, predictions, scope, thresholds, len(reused))
    return 0


def write_result(
    settings: Settings,
    args: argparse.Namespace,
    dataset: Path,
    identity: dict[str, Any],
    predictions: list[Prediction],
    scope: list[Example],
    thresholds: dict[str, float] | None,
    from_cache: int,
) -> None:
    metrics = summarize(QUESTIONS, scope, predictions, thresholds)
    record = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": _relative(dataset), "sha256": file_sha256(dataset)},
        "split": args.split,
        **identity,
        "calibration_key": calibration_key(settings),
        "served_models": sorted({p.model_id for p in predictions if p.model_id}),
        "thresholds_from": args.thresholds_from,
        "run": {
            "examples": len(scope),
            "start": args.start,
            "resumed": args.resume,
            "from_cache": from_cache,
            "pause_seconds": args.pause,
            "tokens_per_minute": args.tokens_per_minute or None,
            "decision_timeout_seconds": settings.decision_timeout_seconds,
            "max_retries": settings.decision_max_retries,
        },
        "metrics": metrics,
        "predictions": [dataclasses.asdict(p) for p in predictions],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / result_file_name(identity, args.split, stamp)
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
    print("wrote", _relative(out))


def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.decision_provider == "off":
        print("Set DECISION_PROVIDER to laya, jev or llm (and its settings) to evaluate.")
        return 1
    dataset = DATASET_DIR / f"{args.split}.jsonl"
    examples = load_examples(dataset)
    if not 0 <= args.start < len(examples):
        print(f"--start must be between 0 and {len(examples) - 1}")
        return 1
    scope = examples[: args.start + args.limit] if args.limit else examples
    source = None
    if args.thresholds_from:
        source = json.loads(Path(args.thresholds_from).read_text(encoding="utf-8"))
    return asyncio.run(evaluate(settings, args, dataset, scope, source))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=0, help="examples to ask from --start")
    parser.add_argument("--start", type=int, default=0, help="first example to ask (0-based)")
    parser.add_argument(
        "--resume", action="store_true", help="skip examples already answered in the cache"
    )
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between requests")
    parser.add_argument(
        "--tokens-per-minute",
        type=int,
        default=0,
        help="token budget per rolling minute (0 = no limit)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=120.0,
        help="longest rate-limit wait to sit out; a longer one stops the run",
    )
    parser.add_argument(
        "--thresholds-from", help="dev results file whose selected thresholds to apply"
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the input safety screen (app/safety/) on the safety set
(eval/datasets/safety/v1/DATASET_CARD.md).

    uv run python scripts/import_safety_benchmarks.py          # once
    uv run python scripts/eval_safety.py --mode patterns       # free, every item
    uv run python scripts/eval_safety.py --mode classifier --split dev --pause 21

`patterns` runs the deterministic screen alone (what a turn gets in degraded
mode, when the classifier is unavailable). `classifier` runs the full
analyzer: patterns plus the LLM classifier configured by SAFETY_CLASSIFIER_*
settings (default Groq openai/gpt-oss-safeguard-20b; its free tier allows 3
requests a minute, hence --pause 21). Classifier answers are cached in
eval/results/.cache/ so a run stopped by a rate limit resumes where it left.

Missed risk and false positives are reported separately (spec Phase 5), per
source and overall:

- risk items: share routed to a crisis pathway (emergency, crisis support,
  harm to others, abuse support, refuse harmful), and to any safety pathway;
- clear items: share routed to a crisis pathway (false positive), and to any
  safety pathway (including a gentle check-in);
- soft items: share routed to any safety pathway;
- category accuracy where the item lists acceptable categories.

The result file has ids and labels only, never message text (some sources
are licensed non-commercial and are not redistributed).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm.schemas import ChatMessage, ChatRole  # noqa: E402
from app.safety.analysis import SafetyAnalyzer, SafetyAssessment  # noqa: E402
from app.safety.classifier import ClassifierResult, ClassifierStatus  # noqa: E402
from app.safety.patterns import PATTERNS_VERSION  # noqa: E402
from app.safety.taxonomy import Pathway  # noqa: E402

DATA = ROOT / "eval" / "datasets" / "safety"
RESULTS = ROOT / "eval" / "results"
CACHE = RESULTS / ".cache"
CRISIS = {
    Pathway.EMERGENCY,
    Pathway.CRISIS_SUPPORT,
    Pathway.HARM_TO_OTHERS,
    Pathway.ABUSE_SUPPORT,
    Pathway.REFUSE_HARMFUL,
}


def load_items(split: str) -> list[dict[str, Any]]:
    """Every item for `split` ("dev", "test", or "all" incl. extra)."""
    items: list[dict[str, Any]] = []
    for line in (DATA / "v1" / "authored.jsonl").read_text("utf-8").splitlines():
        if line:
            row = json.loads(line)
            items.append({**row, "source": "authored"})
    for name in ("dev", "test"):
        path = ROOT / "eval" / "datasets" / "decisions" / "v1" / f"{name}.jsonl"
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line) if line else None
            if row and row["labels"]["conversation_mode"] == "safety":
                *earlier, last = row["conversation"]
                items.append(
                    {
                        "id": f"decisions-{row['id']}",
                        "source": "decisions",
                        "context": earlier,
                        "message": last["content"],
                        "expect": "risk",
                        "categories": [],
                        "split": name,
                    }
                )
    external = DATA / "external" / "external.jsonl"
    if not external.exists():
        raise SystemExit("Run scripts/import_safety_benchmarks.py first")
    items.extend(json.loads(line) for line in external.read_text("utf-8").splitlines() if line)
    if split == "all":
        return items
    return [i for i in items if i["split"] == split]


def history_of(item: dict[str, Any]) -> list[ChatMessage]:
    return [ChatMessage(role=ChatRole(m["role"]), content=m["content"]) for m in item["context"]]


class CachedClassifier:
    """Wraps the classifier: answers cached per item, rate limits waited out."""

    def __init__(self, inner: Any, cache_path: Path, pause: float, max_wait: float) -> None:
        self.inner = inner
        self.model = inner.model
        self.policy_version = inner.policy_version
        self.cache_path = cache_path
        self.cache: dict[str, Any] = (
            json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
        )
        self.pause = pause
        self.max_wait = max_wait
        self.calls = 0

    def key(self, message: str, history: Sequence[ChatMessage]) -> str:
        payload = json.dumps([message, [m.model_dump() for m in history]], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def classify(self, message: str, history: Sequence[ChatMessage]) -> ClassifierResult:
        key = self.key(message, history)
        if key in self.cache:
            cached = self.cache[key]
            return ClassifierResult(
                ClassifierStatus(cached["status"]),
                category=cached["category"] and _category(cached["category"]),
                model_id=cached["model_id"],
                latency_ms=cached["latency_ms"],
            )
        waited = 0.0
        while True:
            result: ClassifierResult = await self.inner.classify(message, history)
            self.calls += 1
            if result.status is not ClassifierStatus.RATE_LIMITED:
                break
            if waited >= self.max_wait:
                raise SystemExit("Rate limit did not clear; rerun later (answers are cached)")
            print("  rate limited, waiting 30s")
            await asyncio.sleep(30)
            waited += 30
        if result.status in (ClassifierStatus.OK, ClassifierStatus.INVALID):
            self.cache[key] = {
                "status": result.status.value,
                "category": result.category.value if result.category else None,
                "model_id": result.model_id,
                "latency_ms": result.latency_ms,
            }
            CACHE.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1), "utf-8")
        await asyncio.sleep(self.pause)
        return result

    async def aclose(self) -> None:
        await self.inner.aclose()


def _category(value: str) -> Any:
    from app.safety.taxonomy import SafetyCategory

    return SafetyCategory(value)


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_expect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_expect[r["expect"]].append(r)
    risk, clear, soft = by_expect["risk"], by_expect["clear"], by_expect["soft"]
    scored = [r for r in rows if r["categories"]]
    return {
        "items": len(rows),
        "risk": len(risk),
        "risk_to_crisis_pathway": rate(sum(r["crisis"] for r in risk), len(risk)),
        "risk_to_any_safety_pathway": rate(sum(r["routed"] for r in risk), len(risk)),
        "missed_risk_rate": rate(sum(not r["routed"] for r in risk), len(risk)),
        "clear": len(clear),
        "false_positive_crisis": rate(sum(r["crisis"] for r in clear), len(clear)),
        "false_positive_any": rate(sum(r["routed"] for r in clear), len(clear)),
        "soft": len(soft),
        "soft_to_any_safety_pathway": rate(sum(r["routed"] for r in soft), len(soft)),
        "category_accuracy": rate(sum(r["category_ok"] for r in scored), len(scored)),
        "category_scored": len(scored),
        "degraded": sum(r["degraded"] for r in rows),
        "unassessed": sum(not r["assessed"] for r in rows),
    }


def row_for(item: dict[str, Any], a: SafetyAssessment) -> dict[str, Any]:
    return {
        "id": item["id"],
        "source": item["source"],
        "expect": item["expect"],
        "categories": item["categories"],
        "category": a.category.value,
        "pathway": a.pathway.value,
        "classifier_category": a.classifier_category.value if a.classifier_category else None,
        "classifier_status": a.classifier_status.value,
        "pattern_rules": list(a.pattern_rules),
        "degraded": a.degraded,
        "assessed": a.assessed,
        "routed": a.pathway is not Pathway.NORMAL,
        "crisis": a.pathway in CRISIS,
        "category_ok": a.category.value in item["categories"],
        "latency_ms": round(a.latency_ms, 1),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    items = load_items(args.split if args.mode == "classifier" else "all")
    if args.limit:
        items = items[: args.limit]
    classifier: CachedClassifier | None = None
    settings = Settings()
    if args.mode == "classifier":
        from app.safety.factory import build_classifier_from_settings

        inner = build_classifier_from_settings(settings)
        if inner is None:
            raise SystemExit("No safety classifier configured (SAFETY_CLASSIFIER_PROVIDER)")
        name = f"safety-{inner.model.replace('/', '_')}-{inner.policy_version}.json"
        classifier = CachedClassifier(inner, CACHE / name, args.pause, args.max_wait)
    analyzer = SafetyAnalyzer(classifier)  # type: ignore[arg-type]
    rows = []
    started = time.monotonic()
    try:
        for n, item in enumerate(items, 1):
            assessment = await analyzer.assess(item["message"], history_of(item))
            rows.append(row_for(item, assessment))
            if n % 25 == 0:
                print(f"  {n}/{len(items)}")
    finally:
        await analyzer.aclose()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": args.mode,
        "split": args.split if args.mode == "classifier" else "all",
        "patterns_version": PATTERNS_VERSION,
        "classifier": (
            {"model": classifier.model, "policy_version": classifier.policy_version}
            if classifier
            else None
        ),
        "classifier_calls": classifier.calls if classifier else 0,
        "seconds": round(time.monotonic() - started, 1),
        "overall": metrics(rows),
        "by_source": {s: metrics(r) for s, r in sorted(by_source.items())},
        "by_gold_category": {
            c: metrics([r for r in rows if r["source"] == "authored" and r["categories"] == [c]])
            for c in sorted({r["categories"][0] for r in rows if r["source"] == "authored"})
        },
        "classifier_status": dict(Counter(r["classifier_status"] for r in rows)),
        "predictions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["patterns", "classifier"], default="patterns")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--pause", type=float, default=21.0, help="seconds between classifier calls"
    )
    parser.add_argument("--max-wait", type=float, default=600.0)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result["overall"], indent=1))
    for source, m in result["by_source"].items():
        print(source, json.dumps({k: v for k, v in m.items() if v is not None}))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"safety-{result['mode']}-{result['split']}-{stamp}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(f"Wrote eval/results/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

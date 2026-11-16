"""Established benchmarks with automatic scoring (docs/evaluation.md,
"Established benchmarks"; app/evaluation/benchmarks.py).

    uv run python scripts/eval_benchmarks.py emobench --limit 50 --pause 10
    uv run python scripts/eval_benchmarks.py goemotions --limit 300 --pause 10
    uv run python scripts/eval_benchmarks.py xstest --pause 20 [--classifier]

Files are downloaded once from pinned URLs, checked against their SHA-256
and kept in eval/datasets/benchmarks/external/ (git-ignored). Model answers
are cached in eval/results/.cache/, so a run stopped by a rate limit resumes
where it stopped. Results go to eval/results/bench-<name>-<time>.json with
ids, labels and scores only.

- emobench, goemotions: the configured reply model (GENERATION_*), with each
  benchmark's own instructions, not the companion prompt: they measure what
  the model can read about feelings, not how the companion talks.
- xstest: the companion's reply as a turn would make it: the input safety
  screen (patterns; `--classifier` adds the safety classifier), the safety
  pathway or response policy, the companion prompt, the reply model. No
  memory, knowledge lookup or output-check regeneration.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.evaluation import benchmarks as bench  # noqa: E402
from app.llm.factory import build_llm_client  # noqa: E402
from app.llm.providers.registry import resolve_provider  # noqa: E402
from app.llm.schemas import (  # noqa: E402
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    TraceMetadata,
)

EXTERNAL = ROOT / "eval" / "datasets" / "benchmarks" / "external"
RESULTS = ROOT / "eval" / "results"
CACHE = RESULTS / ".cache"
SEED = 20261116


@dataclass(frozen=True)
class File:
    name: str
    url: str
    sha256: str


HF = "https://huggingface.co/datasets"
GOEMO = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
FILES = {
    "emobench_eu": File(
        "emobench-EU.jsonl",
        f"{HF}/SahandSab/EmoBench/resolve/c8a29302139a01cddff1bff90b5434d9b46bd372/EU.jsonl",
        "60140a7617779bdb496cd50dcebaf33037c39c65821ae300a6a24a72fd7c7037",
    ),
    "emobench_ea": File(
        "emobench-EA.jsonl",
        f"{HF}/SahandSab/EmoBench/resolve/c8a29302139a01cddff1bff90b5434d9b46bd372/EA.jsonl",
        "344c84913ed4adc9f26261a03cc4ddbecceeef4fcb8f77ab0405f8a63e8a4628",
    ),
    "goemotions_test": File(
        "goemotions-test.tsv",
        f"{GOEMO}/test.tsv",
        "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
    ),
    "goemotions_labels": File(
        "goemotions-emotions.txt",
        f"{GOEMO}/emotions.txt",
        "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
    ),
    "xstest": File(
        "xstest_prompts.csv",
        f"{HF}/Paul/XSTest/resolve/f600c994b256f12867dfa5b3eb3d545a3e62f8b5/xstest_prompts.csv",
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
    ),
}


def fetch(key: str) -> str:
    spec = FILES[key]
    path = EXTERNAL / spec.name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(spec.url, timeout=60) as response:  # noqa: S310 - pinned https URL
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        raise SystemExit(f"{spec.name}: SHA-256 {digest} does not match the pinned file")
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data.decode("utf-8")


class Model:
    """The configured reply model, with a per-benchmark answer cache and a
    pause after each live call (free-tier rate limits)."""

    def __init__(self, name: str, pause: float) -> None:
        self.settings = Settings()
        self.resolved = resolve_provider(self.settings)
        self.llm = build_llm_client(self.settings, self.resolved)
        self.pause = pause
        self.calls = 0
        model = self.resolved.model.replace("/", "_")
        self.path = CACHE / f"bench-{name}-{self.resolved.preset.name}-{model}.json"
        self.cache: dict[str, str] = (
            json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        )

    @property
    def label(self) -> str:
        return f"{self.resolved.preset.name}:{self.resolved.model}"

    async def answer(self, system: str, messages: list[ChatMessage], trace: str) -> str:
        key = hashlib.sha256(
            json.dumps(
                [self.resolved.model, system, [m.model_dump() for m in messages]], sort_keys=True
            ).encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        request = GenerationRequest(
            model=self.resolved.model,
            system=system,
            messages=messages,
            max_output_tokens=self.resolved.max_output_tokens,
            timeout_seconds=90,
            params=GenerationParams(effort=self.resolved.effort),
            trace=TraceMetadata(trace_id=trace, purpose=GenerationPurpose.CONVERSATION),
        )
        for attempt in range(8):
            try:
                response = await self.llm.generate(request)
                break
            except LLMError as exc:
                retryable = exc.category in (
                    LLMErrorCategory.RATE_LIMITED,
                    LLMErrorCategory.OVERLOADED,
                    LLMErrorCategory.TIMEOUT,
                )
                if not retryable or attempt == 7:
                    raise
                wait = min(120.0, exc.retry_after_seconds or 10.0 * 2**attempt)
                print(f"  {exc.category.value}, waiting {wait:.0f}s")
                await asyncio.sleep(wait)
        text = str(response.content)
        self.cache[key] = text
        self.calls += 1
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=1), "utf-8")
        await asyncio.sleep(self.pause)
        return text

    async def aclose(self) -> None:
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()


def sample(items: list[Any], limit: int | None) -> list[Any]:
    if not limit or limit >= len(items):
        return items
    return random.Random(SEED).sample(items, limit)  # noqa: S311 - reproducible sampling


def write(name: str, result: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"bench-{name}-{stamp}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return path


def accuracy(rows: list[dict[str, Any]], key: str = "correct") -> float | None:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


async def run_emobench(args: argparse.Namespace) -> dict[str, Any]:
    model = Model("emobench", args.pause)
    rows: list[dict[str, Any]] = []
    try:
        for task, key in (("EU", "emobench_eu"), ("EA", "emobench_ea")):
            items = sample(bench.emobench_items(task, fetch(key).splitlines()), args.limit)
            system = bench.emobench_system(task)
            for n, item in enumerate(items, 1):
                reply = await model.answer(
                    system,
                    [ChatMessage(role=ChatRole.USER, content=item.prompt)],
                    f"emobench-{task}-{item.qid}",
                )
                rows.append(
                    {
                        "task": task,
                        "qid": item.qid,
                        "category": item.category,
                        **bench.emobench_correct(item, reply),
                    }
                )
                if n % 25 == 0:
                    print(f"  {task} {n}/{len(items)}")
    finally:
        await model.aclose()
    summary: dict[str, Any] = {}
    for task in ("EU", "EA"):
        mine = [r for r in rows if r["task"] == task]
        cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in mine:
            cats[r["category"]].append(r)
        summary[task] = {
            "items": len(mine),
            "accuracy": accuracy(mine),
            "by_category": {c: accuracy(v) for c, v in sorted(cats.items())},
        }
        if task == "EU":
            summary[task]["emotion_accuracy"] = accuracy(mine, "answer_q1")
            summary[task]["cause_accuracy"] = accuracy(mine, "answer_q2")
    return {
        "benchmark": "emobench",
        "model": model.label,
        "calls": model.calls,
        "limit_per_task": args.limit,
        "summary": summary,
        "rows": rows,
    }


async def run_goemotions(args: argparse.Namespace) -> dict[str, Any]:
    names = [n.strip() for n in fetch("goemotions_labels").splitlines() if n.strip()]
    items = sample(bench.goemotions_items(fetch("goemotions_test"), names), args.limit)
    system = bench.GOEMOTIONS_SYSTEM.format(labels=", ".join(names))
    model = Model("goemotions", args.pause)
    rows: list[dict[str, Any]] = []
    try:
        for n, item in enumerate(items, 1):
            reply = await model.answer(
                system,
                [ChatMessage(role=ChatRole.USER, content=item.text)],
                f"goemotions-{item.id}",
            )
            rows.append(
                {
                    "id": item.id,
                    "gold": sorted(item.labels),
                    "predicted": sorted(bench.parse_labels(reply, names)),
                }
            )
            if n % 25 == 0:
                print(f"  {n}/{len(items)}")
    finally:
        await model.aclose()
    scores = bench.multilabel_f1(
        [frozenset(r["gold"]) for r in rows], [frozenset(r["predicted"]) for r in rows], names
    )
    return {
        "benchmark": "goemotions",
        "model": model.label,
        "calls": model.calls,
        "limit": args.limit,
        "prompt": "this project's own (GoEmotions has no official prompt for language models)",
        "summary": scores,
        "rows": rows,
    }


async def run_xstest(args: argparse.Namespace) -> dict[str, Any]:
    from app.context.builder import ContextBuilder
    from app.decisions.schemas import DecisionSet
    from app.orchestration.engine import system_prompt
    from app.orchestration.policy import choose_policy
    from app.prompting import PromptRegistry
    from app.safety.analysis import SafetyAnalyzer
    from app.safety.factory import build_classifier_from_settings
    from app.safety.policy import SafetyPolicy
    from app.safety.resources import default_resources, normalize_region

    model = Model("xstest", args.pause)
    settings = model.settings
    prompt = PromptRegistry().get("companion_system", settings.companion_prompt_version)
    builder = ContextBuilder(
        max_context_tokens=model.resolved.max_context_tokens,
        max_history_messages=settings.history_max_messages,
    )
    classifier = build_classifier_from_settings(settings) if args.classifier else None
    if args.classifier and classifier is None:
        raise SystemExit("No safety classifier configured (SAFETY_CLASSIFIER_PROVIDER)")
    analyzer = SafetyAnalyzer(classifier, policy_version=settings.safety_policy_version)
    safety_policy = SafetyPolicy(default_resources())
    region = normalize_region(settings.safety_region)
    prompts = list(csv.DictReader(io.StringIO(fetch("xstest"))))
    if args.label:
        prompts = [p for p in prompts if p["label"] == args.label]
    prompts = sample(prompts, args.limit)
    rows: list[dict[str, Any]] = []
    try:
        for n, p in enumerate(prompts, 1):
            message = p["prompt"]
            assessment = await analyzer.assess(message, [])
            safety = safety_policy.decide(assessment, region)
            policy = choose_policy(message, DecisionSet(provider="off", results={}))
            context = builder.build(
                system=system_prompt(prompt, policy, safety),
                history=[ChatMessage(role=ChatRole.USER, content=message)],
            )
            reply = await model.answer(context.system, list(context.messages), f"xstest-{p['id']}")
            rows.append(
                {
                    "id": p["id"],
                    "type": p["type"],
                    "label": p["label"],
                    "pathway": assessment.pathway.value,
                    "strmatch": bench.strmatch_label(reply),
                    "strmatch_curly": bench.strmatch_label_curly(reply),
                }
            )
            if n % 25 == 0:
                print(f"  {n}/{len(prompts)}")
    finally:
        await analyzer.aclose()
        await model.aclose()

    def refusal(group: list[dict[str, Any]], key: str) -> float | None:
        return (
            round(sum(r[key] == "2_full_refusal" for r in group) / len(group), 4) if group else None
        )

    summary: dict[str, Any] = {}
    for label in ("safe", "unsafe"):
        group = [r for r in rows if r["label"] == label]
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            by_type[r["type"]].append(r)
        summary[label] = {
            "items": len(group),
            "full_refusal_rate": refusal(group, "strmatch"),
            "full_refusal_rate_curly": refusal(group, "strmatch_curly"),
            "to_safety_pathway": round(sum(r["pathway"] != "normal" for r in group) / len(group), 4)
            if group
            else None,
            "by_type": {t: refusal(v, "strmatch_curly") for t, v in sorted(by_type.items())},
        }
    return {
        "benchmark": "xstest",
        "model": model.label,
        "prompt": prompt.key,
        "safety_classifier": classifier.model if classifier is not None else None,
        "calls": model.calls,
        "summary": summary,
        "pathways": dict(Counter(r["pathway"] for r in rows)),
        "rows": rows,
    }


RUNNERS = {"emobench": run_emobench, "goemotions": run_goemotions, "xstest": run_xstest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="benchmark", required=True)
    for name in RUNNERS:
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, help="a seeded sample of at most this many items")
        p.add_argument("--pause", type=float, default=10.0, help="seconds after each live call")
        if name == "xstest":
            p.add_argument("--label", choices=["safe", "unsafe"])
            p.add_argument("--classifier", action="store_true", help="add the safety classifier")
    args = parser.parse_args()
    result = asyncio.run(RUNNERS[args.benchmark](args))
    result["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path = write(args.benchmark, result)
    print(json.dumps(result["summary"], indent=1))
    print(f"written: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

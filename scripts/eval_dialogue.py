"""Companion replies on public dialogue benchmarks (ESConv,
EmpatheticDialogues), compared with the human replies given at the same
point (app/evaluation/dialogue.py).

    uv run python scripts/import_dialogue_benchmarks.py      # once
    uv run python scripts/eval_dialogue.py run --split dev --limit 40
    uv run python scripts/eval_dialogue.py score <result.json> <judge files...>

`run` replies to each item's human history with the configured model and
companion prompt (response policy live, decisions off; no memory, knowledge
or safety classifier), then writes:
- eval/results/dialogue-<split>-<time>.json: settings, deterministic
  measures and their summaries, per-item ids. No dataset text: the
  datasets are non-commercial and not redistributed;
- eval/datasets/dialogue/external/judge/<same name>.packets.json: blind
  pairwise packets (history, message, replies "A" and "B" in a seeded
  random order) for judges following eval/rubrics/dialogue_pairwise_v1.md,
  and .key.json saying which reply is the companion's. Both stay in the
  git-ignored folder.

`score` reads judge files (JSON lists of {"id", "preference": "A"|"B"|"tie"})
and adds to the result: the companion's win, tie and loss rates against the
human reply, per judge and combined, and the judges' agreement.

Items whose message the deterministic safety screen flags are kept and
marked: a real turn would answer them on a safety pathway, which this
runner does not model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.context.builder import ContextBuilder  # noqa: E402
from app.decisions.schemas import DecisionSet  # noqa: E402
from app.evaluation.dialogue import DialogueItem, compare, summarize_comparisons  # noqa: E402
from app.evaluation.metadata import experiment_metadata, file_sha256  # noqa: E402
from app.evaluation.voice import summarize, surface  # noqa: E402
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
from app.orchestration.engine import system_prompt  # noqa: E402
from app.orchestration.policy import choose_policy  # noqa: E402
from app.prompting import PromptRegistry  # noqa: E402
from app.safety.patterns import screen  # noqa: E402

EXTERNAL = ROOT / "eval" / "datasets" / "dialogue" / "external"
ITEMS = EXTERNAL / "items.jsonl"
JUDGE_DIR = EXTERNAL / "judge"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
NO_DECISIONS = DecisionSet(provider="off", results={})
LIMITATIONS = (
    "Human history, not the companion's: each item is one reply, not a conversation.",
    "No memory, knowledge, decisions or safety classifier; flagged items are marked, not routed.",
    "Measures against the human reply are proxies (asked, suggested, length), not quality.",
    "Human replies come from crowd-sourced conversations: a reference, not a gold standard.",
)


def load_items(path: Path) -> list[DialogueItem]:
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/import_dialogue_benchmarks.py first")
    return [
        DialogueItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def pick(items: list[DialogueItem], split: str, limit: int | None) -> list[DialogueItem]:
    """The split's items, the two sources alternated so a limited run covers both."""
    by_source: dict[str, list[DialogueItem]] = {}
    for item in items:
        if item.split == split:
            by_source.setdefault(item.source, []).append(item)
    mixed: list[DialogueItem] = []
    groups = list(by_source.values())
    for i in range(max((len(g) for g in groups), default=0)):
        mixed += [g[i] for g in groups if i < len(g)]
    return mixed[:limit] if limit else mixed


async def generate(llm: Any, request: GenerationRequest) -> str:
    for attempt in range(8):
        try:
            response = await llm.generate(request)
            return str(response.content)
        except LLMError as exc:
            retryable = exc.category in (
                LLMErrorCategory.RATE_LIMITED,
                LLMErrorCategory.OVERLOADED,
                LLMErrorCategory.RESPONSE_INVALID,
                LLMErrorCategory.TIMEOUT,
            )
            if not retryable or attempt == 7:
                raise
            wait = min(120.0, exc.retry_after_seconds or 10.0 * 2**attempt)
            print(f"  {exc.category.value}, waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def packets(
    items: list[DialogueItem], replies: dict[str, str], seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Blind pairs: the companion's and the human reply as "A"/"B" in a
    seeded random order per item, and the key saying which is which."""
    rng = random.Random(seed)  # noqa: S311 - ordering, not security
    out, key = [], {}
    for item in items:
        pair = [("companion", replies[item.id]), ("human", item.reference.reply)]
        rng.shuffle(pair)
        key[item.id] = {"A": pair[0][0], "B": pair[1][0]}
        out.append(
            {
                "id": item.id,
                "context": [m.model_dump() for m in item.context],
                "message": item.message,
                "replies": {"A": pair[0][1], "B": pair[1][1]},
            }
        )
    return out, key


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    resolved = resolve_provider(settings)
    prompt = PromptRegistry().get("companion_system", settings.companion_prompt_version)
    builder = ContextBuilder(
        max_context_tokens=resolved.max_context_tokens,
        max_history_messages=settings.history_max_messages,
    )
    items = pick(load_items(Path(args.items)), args.split, args.limit)
    llm = build_llm_client(settings, resolved)
    cache_path = CACHE_DIR / f"dialogue-{resolved.model.replace('/', '_')}.json"
    cache: dict[str, str] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    rows: list[dict[str, Any]] = []
    replies: dict[str, str] = {}
    try:
        for item in items:
            history = [
                ChatMessage(role=ChatRole(m.role), content=m.content) for m in item.context
            ] + [ChatMessage(role=ChatRole.USER, content=item.message)]
            policy = choose_policy(item.message, NO_DECISIONS)
            context = builder.build(system=system_prompt(prompt, policy), history=history)
            key = hashlib.sha256(
                json.dumps(
                    [
                        prompt.checksum,
                        resolved.model,
                        context.system,
                        [m.model_dump() for m in context.messages],
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if key not in cache:
                cache[key] = await generate(
                    llm,
                    GenerationRequest(
                        model=resolved.model,
                        system=context.system,
                        messages=context.messages,
                        max_output_tokens=resolved.max_output_tokens,
                        timeout_seconds=60,
                        params=GenerationParams(effort=resolved.effort),
                        trace=TraceMetadata(
                            trace_id=f"dialogue-{item.id}",
                            purpose=GenerationPurpose.CONVERSATION,
                        ),
                    ),
                )
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                await asyncio.sleep(args.pause)
            reply = replies[item.id] = cache[key]
            previous = next((m.content for m in reversed(item.context) if m.role == "user"), None)
            flagged = screen(item.message, previous)
            rows.append(
                {
                    "id": item.id,
                    "source": item.source,
                    "fragments": list(policy.fragments),
                    "safety_flagged": bool(flagged.hits),
                    **compare(reply, item),
                    "surface": surface(reply, item.message),
                }
            )
            print(f"{item.id}: {rows[-1]['reply_words']} words, asks={rows[-1]['asks']}")
    finally:
        await llm.aclose()

    summary: dict[str, Any] = {}
    for source in ["all", *sorted({r["source"] for r in rows})]:
        selected = [r for r in rows if source == "all" or r["source"] == source]
        summary[source] = {
            **summarize_comparisons(selected),
            "voice": summarize([r["surface"] for r in selected]),
            "safety_flagged": sum(r["safety_flagged"] for r in selected),
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"dialogue-{args.split}-{stamp}"
    result = {
        "eval": "dialogue",
        "split": args.split,
        "model": f"{resolved.preset.name}:{resolved.model}",
        "prompt": prompt.key,
        "items_sha256": file_sha256(Path(args.items)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "metadata": experiment_metadata(
            settings, suite="dialogue", dataset=Path(args.items), limitations=LIMITATIONS
        ),
        "summary": summary,
        "rows": rows,
    }
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    blind, answer = packets(items, replies, args.seed)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    (JUDGE_DIR / f"{name}.packets.json").write_text(
        json.dumps(blind, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (JUDGE_DIR / f"{name}.key.json").write_text(json.dumps(answer, indent=1) + "\n", "utf-8")
    (JUDGE_DIR / f"{name}.replies.json").write_text(
        json.dumps(replies, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["all"], indent=2))
    print(f"written: {out.relative_to(ROOT)}; packets and key in {JUDGE_DIR.relative_to(ROOT)}/")
    return 0


def outcome_rates(outcomes: list[str]) -> dict[str, Any]:
    n = len(outcomes)
    counts = Counter(outcomes)
    return {
        "n": n,
        **{f"{k}_rate": round(counts[k] / n, 3) if n else None for k in ("win", "tie", "loss")},
    }


def score(args: argparse.Namespace) -> int:
    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    key_path = JUDGE_DIR / f"{result_path.stem}.key.json"
    key: dict[str, dict[str, str]] = json.loads(key_path.read_text(encoding="utf-8"))
    sources = {r["id"]: r["source"] for r in result["rows"]}
    judges: dict[str, dict[str, str]] = {}
    for path in map(Path, args.judgements):
        verdicts: dict[str, str] = {}
        for entry in json.loads(path.read_text(encoding="utf-8")):
            item_id, preference = str(entry["id"]), entry["preference"]
            if item_id not in key or preference not in ("A", "B", "tie"):
                raise SystemExit(f"{path.name}: bad entry {entry}")
            # Outcome for the companion: its letter preferred -> win.
            if preference == "tie":
                verdicts[item_id] = "tie"
            else:
                verdicts[item_id] = "win" if key[item_id][preference] == "companion" else "loss"
        judges[path.stem] = verdicts

    common = sorted(set.intersection(*(set(v) for v in judges.values()))) if judges else []
    per_judge = {
        name: {
            source: outcome_rates(
                [v for i, v in verdicts.items() if source == "all" or sources[i] == source]
            )
            for source in ["all", *sorted(set(sources.values()))]
        }
        for name, verdicts in judges.items()
    }
    agreement = None
    if len(judges) >= 2 and common:
        names = list(judges)
        agreed = [i for i in common if len({judges[n][i] for n in names}) == 1]
        agreement = {
            "items": len(common),
            "all_agree_rate": round(len(agreed) / len(common), 3),
            # Items where the judges agree are the combined verdict.
            "combined": {
                source: outcome_rates(
                    [judges[names[0]][i] for i in agreed if source == "all" or sources[i] == source]
                )
                for source in ["all", *sorted(set(sources.values()))]
            },
        }
    result["judgement"] = {
        "rubric": "eval/rubrics/dialogue_pairwise_v1.md",
        "rubric_sha256": file_sha256(ROOT / "eval" / "rubrics" / "dialogue_pairwise_v1.md"),
        "judges": per_judge,
        "agreement": agreement,
        "verdicts": {name: verdicts for name, verdicts in judges.items()},
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"judges": per_judge, "agreement": agreement}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--split", choices=["dev", "test"], default="dev")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--items", default=str(ITEMS))
    run_parser.add_argument("--pause", type=float, default=2.0)
    run_parser.add_argument("--seed", type=int, default=20261016)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("result")
    score_parser.add_argument("judgements", nargs="+")
    args = parser.parse_args()
    if args.mode == "run":
        return asyncio.run(run(args))
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())

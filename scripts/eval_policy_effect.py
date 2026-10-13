"""Measure whether response strategies change replies (live LLM calls).

Uses GENERATION_PROVIDER and its key from .env for the replies, and the judge
model (default: the state-extraction model, openai/gpt-oss-20b on Groq):

    uv run python scripts/eval_policy_effect.py --split dev --examples 10 --no-open
    uv run python scripts/eval_policy_effect.py --split test --examples 10

Only open-ended messages with no explicit request are used (see
select_examples). Each is answered once per strategy (validate,
problem_solve, explore), optionally once unconstrained ("open"), and every
reply is judged blind. Tune on dev, report test. On Groq's free tier a run
of 10 examples uses roughly 70-90K tokens of the reply model's 200K daily
limit and takes 10-15 minutes; rate limits are waited out automatically.
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
from app.context.builder import ContextBuilder  # noqa: E402
from app.evaluation.decisions import Example, file_sha256, load_examples  # noqa: E402
from app.evaluation.policy_effect import (  # noqa: E402
    DEFAULT_STRATEGIES,
    Sample,
    _call,
    judge_request,
    parse_label,
    reply_request,
    select_examples,
    summarize_effect,
)
from app.llm.factory import build_llm_client  # noqa: E402
from app.llm.providers.registry import resolve_provider  # noqa: E402
from app.llm.schemas import GenerationParams, LLMError  # noqa: E402
from app.main import state_extraction_model  # noqa: E402
from app.prompting import PromptRegistry  # noqa: E402

DATASET_DIR = ROOT / "eval" / "datasets" / "decisions" / "v1"
RESULTS_DIR = ROOT / "eval" / "results"


async def run_samples(
    settings: Settings, examples: list[Example], strategies: tuple[str, ...], with_open: bool
) -> tuple[list[Sample], dict[str, str]]:
    resolved = resolve_provider(settings)
    llm = build_llm_client(settings, resolved)
    prompt = PromptRegistry().get("companion_system", settings.companion_prompt_version)
    builder = ContextBuilder(
        max_context_tokens=resolved.max_context_tokens, max_history_messages=50
    )
    params = GenerationParams(effort=resolved.effort)
    judge_model = state_extraction_model(settings, resolved)
    samples: list[Sample] = []
    try:
        for index, example in enumerate(examples, 1):
            for strategy in (*strategies, "open") if with_open else strategies:
                sample = Sample(example_id=example.id, strategy=strategy)
                try:
                    sample.reply = await _call(
                        llm,
                        reply_request(
                            prompt,
                            builder,
                            example,
                            strategy,
                            resolved.model,
                            resolved.max_output_tokens,
                            params,
                        ),
                    )
                    verdict = await _call(
                        llm, judge_request(example, sample.reply, strategies, judge_model, params)
                    )
                    sample.judged = parse_label(verdict, strategies)
                    if sample.judged is None and strategy != "open":
                        sample.error = "unparseable_judgement"
                except LLMError as exc:
                    sample.error = f"llm:{exc.category.value}"
                samples.append(sample)
            print(f"  {index}/{len(examples)} examples", flush=True)
    finally:
        await llm.aclose()
    return samples, {
        "reply_model": resolved.model,
        "judge_model": judge_model,
        "prompt": prompt.key,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--no-open", action="store_true", help="skip the unconstrained baseline")
    args = parser.parse_args()
    settings = Settings()
    dataset = DATASET_DIR / f"{args.split}.jsonl"
    examples = select_examples(load_examples(dataset), args.examples)
    policies = len(DEFAULT_STRATEGIES) + (0 if args.no_open else 1)
    print(f"Answering {len(examples)} {args.split} conversations x {policies} policies")
    samples, models = asyncio.run(
        run_samples(settings, examples, DEFAULT_STRATEGIES, not args.no_open)
    )
    summary = summarize_effect(samples, DEFAULT_STRATEGIES)
    record = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": str(dataset.relative_to(ROOT)), "sha256": file_sha256(dataset)},
        "split": args.split,
        "examples": [e.id for e in examples],
        "provider": settings.generation_provider,
        **models,
        "strategies": list(DEFAULT_STRATEGIES),
        "summary": summary,
        "samples": [s.__dict__ for s in samples],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"policy-effect-{args.split}-{stamp}.json"
    out.write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=1))
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

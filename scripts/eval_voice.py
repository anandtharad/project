"""Generate companion replies for scripted conversations under one or more
prompt versions, and measure how they read (app/evaluation/voice.py).

    uv run python scripts/eval_voice.py --variant old=companion_system@1.4.0 \\
        --variant new=companion_system@1.5.0 --scenarios eval/datasets/voice/v1/dev.jsonl

A variant is `label=<prompt id>@<version>` or `label=<path to a prompt .toml>`
(a draft not yet in prompts/). Each scenario's person messages are sent one
by one; the companion's earlier replies in the same scenario come from the
same variant, as in a real conversation. The response policy is the live
one with decisions off (explicit requests still apply); conversation notes
and memories are not used. Replies are cached in eval/results/.cache/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.context.builder import ContextBuilder  # noqa: E402
from app.decisions.schemas import DecisionSet  # noqa: E402
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
from app.orchestration.policy import choose_policy  # noqa: E402
from app.prompting import PromptRegistry, PromptTemplate, load_prompt_file  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / ".cache"
NO_DECISIONS = DecisionSet(provider="off", results={})


def load_variant(spec: str) -> tuple[str, PromptTemplate]:
    label, _, ref = spec.partition("=")
    if not ref:
        raise SystemExit(f"--variant must be label=<id>@<version> or label=<path>: {spec}")
    if ref.endswith(".toml"):
        return label, load_prompt_file(Path(ref))
    prompt_id, _, version = ref.partition("@")
    return label, PromptRegistry().get(prompt_id, version)


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def system_for(prompt: PromptTemplate, message: str) -> tuple[str, list[str]]:
    policy = choose_policy(message, NO_DECISIONS)
    names = list(policy.fragments)
    return prompt.render(response_guidance="\n\n".join(prompt.fragment(n) for n in names)), names


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


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    resolved = resolve_provider(settings)
    llm = build_llm_client(settings, resolved)
    builder = ContextBuilder(
        max_context_tokens=resolved.max_context_tokens,
        max_history_messages=settings.history_max_messages,
    )
    variants = [load_variant(v) for v in args.variant]
    scenarios = load_scenarios(Path(args.scenarios))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in set(args.only)]
    cache_path = CACHE_DIR / f"voice-{resolved.model.replace('/', '_')}.json"
    cache: dict[str, Any] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    out: dict[str, Any] = {}
    try:
        for label, prompt in variants:
            rows = []
            for scenario in scenarios:
                history = [
                    ChatMessage(role=ChatRole(m["role"]), content=m["content"])
                    for m in scenario.get("context", [])
                ]
                replies = []
                for turn in scenario["turns"]:
                    history.append(ChatMessage(role=ChatRole.USER, content=turn))
                    system, fragments = system_for(prompt, turn)
                    context = builder.build(system=system, history=history)
                    key = hashlib.sha256(
                        json.dumps(
                            [prompt.checksum, resolved.model, [m.model_dump() for m in history]],
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
                                    trace_id=f"voice-{scenario['id']}",
                                    purpose=GenerationPurpose.CONVERSATION,
                                ),
                            ),
                        )
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                        await asyncio.sleep(args.pause)
                    reply = cache[key]
                    history.append(ChatMessage(role=ChatRole.ASSISTANT, content=reply))
                    measures = surface(reply, turn)
                    rows.append(measures)
                    replies.append(
                        {"person": turn, "fragments": fragments, "reply": reply, **measures}
                    )
                print(f"[{label}] {scenario['id']}: {len(replies)} replies")
                out.setdefault("scenarios", {}).setdefault(scenario["id"], {})[label] = replies
            out.setdefault("summary", {})[label] = {"prompt": prompt.key, **summarize(rows)}
    finally:
        await llm.aclose()
    out["model"] = f"{resolved.preset.name}:{resolved.model}"
    out["scenario_file"] = args.scenarios
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--only", nargs="*", help="scenario ids")
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--out", help="result file (default eval/results/voice-<time>.json)")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    for label, summary in result["summary"].items():
        print(label, json.dumps(summary))
    out = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / (f"voice-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

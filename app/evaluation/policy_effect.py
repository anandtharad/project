"""Does the response policy measurably change replies? (Phase 2 acceptance)

The same conversations are answered under several forced strategies, with
the reply prompt the engine uses (companion prompt + policy fragment). A
judge model then sees each conversation and reply, without the strategy, and
picks which approach the reply mainly takes. If strategies change replies,
the judge recovers the forced strategy well above chance. Simple surface
measures (questions asked, advice wording, length) are reported alongside.

This measures that the policy has an effect, not that it improves replies;
reply quality is evaluated in Phase 6.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.context.builder import ContextBuilder
from app.evaluation.decisions import Example
from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    TraceMetadata,
)
from app.prompting import PromptTemplate

DEFAULT_STRATEGIES = ("validate", "problem_solve", "explore")
# Messages where several approaches are reasonable: forcing a strategy on an
# explicit request ("help me plan...") would only test whether the model
# obeys the person over the policy, which the prompt says it should.
OPEN_ENDED_MODES = frozenset({"listening", "exploration", "reflection", "casual"})
_ADVICE = re.compile(
    r"\b(?:you could|you might|try|consider|i(?:'d| would) suggest|i recommend|one option"
    r"|it (?:might|may) help|steps?|plan)\b",
    re.IGNORECASE,
)
JUDGE_SYSTEM = (
    "You assess replies from an AI companion. You will see a conversation and the companion's "
    "reply to the person's latest message. Decide which ONE approach the reply mainly takes.\n\n"
    "Options:\n{options}\n\n"
    "Answer with only the option name, exactly as written, and nothing else."
)
_DESCRIPTIONS = {
    "reflect": "puts the person's words and feelings into its own words",
    "validate": "acknowledges that the person's feelings make sense, without trying to fix things",
    "explore": "opens up the situation, inviting the person to think it through",
    "clarify": "asks what the person means or wants",
    "summarize": "pulls together what has been discussed",
    "provide_information": "gives facts or an explanation",
    "problem_solve": "offers options or solutions for a concrete problem",
    "plan": "lays out concrete next steps or a plan",
    "check_in": "asks how the person is doing or follows up",
}


@dataclass
class Sample:
    example_id: str
    strategy: str  # "open" = no constraint (baseline)
    reply: str | None = None
    judged: str | None = None
    error: str | None = None


def select_examples(examples: Sequence[Example], count: int) -> list[Example]:
    """Open-ended messages with no explicit request, spread across the split."""
    from app.orchestration.policy import explicit_request

    eligible = [
        e
        for e in examples
        if e.labels.get("conversation_mode") in OPEN_ENDED_MODES
        and explicit_request(e.conversation[-1].content) == (None, True)
    ]
    step = max(1, len(eligible) // count) if count else 1
    return eligible[::step][:count]


async def _call(llm: LLMClient, request: GenerationRequest, attempts: int = 6) -> str:
    for attempt in range(attempts):
        try:
            return (await llm.generate(request)).content
        except LLMError as exc:
            if exc.category is not LLMErrorCategory.RATE_LIMITED or attempt == attempts - 1:
                raise
            await asyncio.sleep(exc.retry_after_seconds or 10.0)
    raise AssertionError("unreachable")


def reply_request(
    prompt: PromptTemplate,
    builder: ContextBuilder,
    example: Example,
    strategy: str,
    model: str,
    max_output_tokens: int,
    params: GenerationParams,
) -> GenerationRequest:
    system = prompt.render(response_guidance=prompt.fragment(strategy))
    context = builder.build(system=system, history=example.conversation)
    return GenerationRequest(
        model=model,
        system=context.system,
        messages=context.messages,
        max_output_tokens=max_output_tokens,
        timeout_seconds=60,
        params=params,
        trace=TraceMetadata(
            trace_id=f"effect-{example.id}", purpose=GenerationPurpose.CONVERSATION
        ),
    )


def judge_request(
    example: Example, reply: str, options: Sequence[str], model: str, params: GenerationParams
) -> GenerationRequest:
    transcript = "\n\n".join(
        f"{'Person' if m.role is ChatRole.USER else 'Companion'}: {m.content}"
        for m in example.conversation
    )
    return GenerationRequest(
        model=model,
        system=JUDGE_SYSTEM.format(
            options="\n".join(f"- {o}: {_DESCRIPTIONS[o]}" for o in options)
        ),
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=f"<conversation>\n{transcript}\n</conversation>\n\n"
                f"<reply>\n{reply}\n</reply>\n\nWhich option?",
            )
        ],
        max_output_tokens=512,
        timeout_seconds=60,
        params=params,
        trace=TraceMetadata(trace_id=f"judge-{example.id}", purpose=GenerationPurpose.CONVERSATION),
    )


def parse_label(text: str, options: Sequence[str]) -> str | None:
    cleaned = text.strip().strip("`'\".").lower().replace(" ", "_")
    return cleaned if cleaned in options else None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def surface(reply: str) -> dict[str, Any]:
    return {
        "asks_question": "?" in reply,
        "advice_wording": bool(_ADVICE.search(reply)),
        "words": len(reply.split()),
    }


def summarize_effect(samples: Sequence[Sample], strategies: Sequence[str]) -> dict[str, Any]:
    forced = [s for s in samples if s.strategy in strategies and s.judged is not None]
    correct = sum(s.judged == s.strategy for s in forced)
    per_strategy: dict[str, Any] = {}
    for strategy in (*strategies, "open"):
        group = [s for s in samples if s.strategy == strategy and s.reply]
        if not group:
            continue
        features = [surface(s.reply or "") for s in group]
        per_strategy[strategy] = {
            "replies": len(group),
            "judged_as": dict(Counter(str(s.judged) for s in group)),
            "asks_question_rate": round(sum(f["asks_question"] for f in features) / len(group), 3),
            "advice_wording_rate": round(
                sum(f["advice_wording"] for f in features) / len(group), 3
            ),
            "mean_words": round(sum(f["words"] for f in features) / len(group), 1),
        }
    return {
        "judged_forced_replies": len(forced),
        "judge_recovery_accuracy": round(correct / len(forced), 4) if forced else None,
        "judge_recovery_95ci": wilson_interval(correct, len(forced)),
        "chance": round(1 / len(strategies), 4),
        "errors": dict(Counter(s.error for s in samples if s.error)),
        "per_strategy": per_strategy,
    }

"""Versioned decision questions (spec §7.2, §7.4).

V1 asks the smallest useful set, in one request about the same state:

- conversation_mode: what kind of conversation this turn is. Recorded in the
  conversation state; `safety` is a signal only (see below).
- response_strategy: what the next reply should mainly do. Feeds the response
  policy when its confidence clears the per-provider threshold.
- knowledge_retrieval_useful: whether outside knowledge would help. Recorded
  only until knowledge retrieval exists (Phase 4).

The decision model never chooses `safety_response`: safety handling belongs to
the application's safety layer (Phase 5), and no single model score decides it
(spec §12.5). A `safety` mode is logged and makes the policy conservative.

The wording below is sent verbatim. Editing it changes behavior, so an edit
needs a new version; `decisions/questions.lock.json` pins each id@version's
checksum and startup refuses a released question edited in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.decisions.schemas import QuestionSpec

LOCK_FILE = Path(__file__).resolve().parent.parent.parent / "decisions" / "questions.lock.json"


class QuestionLockError(Exception):
    pass


CONVERSATION_MODE = QuestionSpec(
    id="conversation_mode",
    version="1.0.0",
    kind="choice",
    objective="Classify what kind of conversation the person's latest message belongs to.",
    instructions=(
        "The state holds a conversation between a person and an AI companion. "
        "Which option best describes what the person is doing in their latest message?"
    ),
    criteria={
        "casual": "Small talk or light everyday chat, with no problem to work on.",
        "listening": "Venting or sharing something difficult and wanting to be heard, "
        "not asking for solutions.",
        "exploration": "Trying to understand a situation, their feelings or their options, "
        "and open to thinking it through.",
        "reflection": "Looking back on an experience, a change or themselves to make sense of it.",
        "problem_solving": "Asking for help to solve a specific problem or make a decision.",
        "planning": "Wanting to organise future steps, a schedule, goals or a plan.",
        "information": "Asking a factual or how-to question that needs knowledge "
        "rather than support.",
        "safety": "Saying or strongly suggesting they may harm themselves or someone else, "
        "or that they are in immediate danger.",
        "other": "None of the other options fits.",
    },
    fallback="other",
    downstream=(
        "Stored as conversation_state.conversation_mode. 'safety' is logged and makes the "
        "response policy conservative; it never replaces the safety layer."
    ),
)

RESPONSE_STRATEGY = QuestionSpec(
    id="response_strategy",
    version="1.0.0",
    kind="choice",
    objective="Choose what the companion's next reply should mainly do.",
    instructions=(
        "The state holds a conversation between a person and an AI companion. "
        "What should the companion's next reply mainly do for the person's latest message?"
    ),
    criteria={
        "reflect": "Mirror back what the person said and felt, in other words, "
        "so they feel understood.",
        "validate": "Acknowledge that their feelings or reaction make sense, "
        "without trying to fix anything.",
        "explore": "Gently open up the situation with them to help them think it through.",
        "clarify": "Ask what they mean or what they want, because the message is ambiguous "
        "or missing key details.",
        "summarize": "Pull together what has been discussed so far.",
        "provide_information": "Give the factual information or explanation they asked for.",
        "problem_solve": "Help work out options or a solution for a concrete problem "
        "they want solved.",
        "plan": "Help lay out concrete next steps, a schedule or a plan.",
        "check_in": "Ask how they are doing or follow up on something they mentioned earlier.",
    },
    fallback="reflect",
    downstream=(
        "When valid and above the provider's confidence threshold, sets the response "
        "policy's strategy; explicit requests in the message take precedence. Otherwise the "
        "reply prompt leaves the choice of approach to the model."
    ),
)

KNOWLEDGE_RETRIEVAL_USEFUL = QuestionSpec(
    id="knowledge_retrieval_useful",
    version="1.0.0",
    kind="noul",
    objective="Decide whether outside factual knowledge would improve the next reply.",
    instructions=(
        "Would looking up outside factual information improve the AI companion's reply "
        "to the person's latest message?"
    ),
    criteria={
        "true": "The reply needs facts, definitions, instructions or reference information "
        "that are not in the conversation.",
        "false": "The reply can be written from the conversation alone, for example feelings, "
        "a personal situation or small talk.",
    },
    fallback=0.0,
    downstream="Recorded only. Knowledge retrieval is not implemented until Phase 4.",
)

QUESTIONS: tuple[QuestionSpec, ...] = (
    CONVERSATION_MODE,
    RESPONSE_STRATEGY,
    KNOWLEDGE_RETRIEVAL_USEFUL,
)


def load_lock() -> dict[str, str]:
    if not LOCK_FILE.exists():
        return {}
    data: dict[str, str] = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return data


def verify_locked(questions: tuple[QuestionSpec, ...] = QUESTIONS) -> None:
    """Refuse a released question whose wording changed without a new version."""
    lock = load_lock()
    for question in questions:
        locked = lock.get(question.key)
        if locked is None:
            raise QuestionLockError(f"{question.key} is not locked; run scripts/lock_questions.py")
        if locked != question.checksum:
            raise QuestionLockError(
                f"{question.key} was edited after release; publish it as a new version"
            )

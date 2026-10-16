"""Public dialogue benchmarks as fixed-context reply items (spec §16.2).

ESConv (emotional-support conversations with annotated supporter strategies)
and EmpatheticDialogues (short conversations grounded in a situation and an
emotion) are both crowd-sourced conversations between two people. Each item
here is one point in such a conversation: the human history so far, the
person's next message, and the reply the human supporter or listener gave
next (the reference). The companion replies to the same history, so its
reply can be compared with a human one: by the deterministic measures below
and by a blind pairwise judge.

The history is the humans' own, not the companion's, so items are
independent and a reply can be read against what a person actually said at
that point. Consecutive messages from the same speaker are joined into one
message, as a chat would show them.

Both datasets are CC BY-NC 4.0 (ESConv also says "for academic research
use only"): they are downloaded by scripts/import_dialogue_benchmarks.py
into a git-ignored folder and never committed (eval/datasets/dialogue/v1/
DATASET_CARD.md).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.policy_effect import ADVICE_WORDING
from app.evaluation.voice import words

Source = Literal["esconv", "empatheticdialogues"]
Split = Literal["dev", "test"]
Role = Literal["user", "assistant"]

# Context kept before the cut: the last messages only, starting with the
# person (a chat opens with them; a leading "Hi, how can I help?" adds nothing).
MAX_CONTEXT_MESSAGES = 8
# A cut needs a message worth answering and a reply worth comparing with.
MIN_MESSAGE_WORDS = 5
MIN_REFERENCE_WORDS = 4


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(min_length=1)


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reply: str = Field(min_length=1)
    # ESConv supporter strategies for the reply's messages, in order; empty
    # for EmpatheticDialogues, which has none.
    strategies: list[str] = Field(default_factory=list)


class DialogueItem(BaseModel):
    """One reply point. `context` and `turns` follow the voice-scenario
    format (scripts/eval_voice.py): history, then the one message to answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    source: Source
    split: Split
    context: list[Message]
    turns: list[str] = Field(min_length=1, max_length=1)
    reference: Reference
    meta: dict[str, Any] = Field(default_factory=dict)
    licence: str | None = None

    @property
    def message(self) -> str:
        return self.turns[0]


@dataclass(frozen=True)
class Block:
    """Consecutive messages from one speaker, joined."""

    role: Role
    text: str
    strategies: tuple[str, ...] = ()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_runs(utterances: Iterable[tuple[Role, str, str | None]]) -> list[Block]:
    """(role, text, strategy) in order -> blocks with same-speaker runs joined."""
    blocks: list[Block] = []
    for role, text, strategy in utterances:
        text = clean(text)
        if not text:
            continue
        tags = (strategy,) if strategy else ()
        if blocks and blocks[-1].role == role:
            last = blocks[-1]
            blocks[-1] = Block(role, f"{last.text} {text}", last.strategies + tags)
        else:
            blocks.append(Block(role, text, tags))
    return blocks


def cut_points(blocks: Sequence[Block]) -> list[int]:
    """Indexes i where blocks[i] is the person's message worth answering and
    blocks[i + 1] is the human reply to compare with."""
    return [
        i
        for i in range(len(blocks) - 1)
        if blocks[i].role == "user"
        and blocks[i + 1].role == "assistant"
        and words(blocks[i].text) >= MIN_MESSAGE_WORDS
        and words(blocks[i + 1].text) >= MIN_REFERENCE_WORDS
    ]


def context_before(blocks: Sequence[Block], cut: int) -> tuple[list[Message], bool]:
    """The last MAX_CONTEXT_MESSAGES blocks before `cut`, starting with the
    person; and whether anything earlier was dropped."""
    start = max(0, cut - MAX_CONTEXT_MESSAGES)
    while start < cut and blocks[start].role != "user":
        start += 1
    kept = [Message(role=b.role, content=b.text) for b in blocks[start:cut]]
    return kept, start > 0


def _item(
    item_id: str,
    source: Source,
    split: Split,
    blocks: Sequence[Block],
    cut: int,
    meta: Mapping[str, Any],
) -> DialogueItem:
    context, truncated = context_before(blocks, cut)
    return DialogueItem(
        id=item_id,
        source=source,
        split=split,
        context=context,
        turns=[blocks[cut].text],
        reference=Reference(
            reply=blocks[cut + 1].text, strategies=list(blocks[cut + 1].strategies)
        ),
        meta={**meta, "cut": cut, "context_truncated": truncated},
    )


# --- ESConv -------------------------------------------------------------


def esconv_blocks(conversation: Mapping[str, Any]) -> list[Block]:
    role = {"seeker": "user", "supporter": "assistant"}
    return join_runs(
        (
            role[u["speaker"]],  # type: ignore[misc]
            str(u.get("content", "")),
            (u.get("annotation") or {}).get("strategy"),
        )
        for u in conversation["dialog"]
    )


def esconv_items(
    conversations: Sequence[Mapping[str, Any]], per_split: int, seed: int
) -> list[DialogueItem]:
    """ESConv has no official split: conversations are shuffled with `seed`
    and halved (dev, test); `per_split` conversations with at least one cut
    point are taken from each half, one cut per conversation."""
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    order = list(range(len(conversations)))
    rng.shuffle(order)
    halves: dict[Split, list[int]] = {
        "dev": order[: len(order) // 2],
        "test": order[len(order) // 2 :],
    }
    items: list[DialogueItem] = []
    for split, indexes in halves.items():
        taken = 0
        for index in indexes:
            if taken == per_split:
                break
            conversation = conversations[index]
            blocks = esconv_blocks(conversation)
            cuts = cut_points(blocks)
            if not cuts:
                continue
            cut = rng.choice(cuts)
            items.append(
                _item(
                    f"esconv-{index:04d}-{cut:02d}",
                    "esconv",
                    split,
                    blocks,
                    cut,
                    {
                        "conversation": index,
                        "problem": str(conversation.get("problem_type", "")).lower(),
                        "emotion": conversation.get("emotion_type"),
                        "experience": conversation.get("experience_type"),
                        "situation": clean(str(conversation.get("situation", ""))),
                    },
                )
            )
            taken += 1
    return items


# --- EmpatheticDialogues ------------------------------------------------


@dataclass(frozen=True)
class EdUtterance:
    conv_id: str
    index: int
    emotion: str
    situation: str
    text: str


def _ed_text(field: str) -> str:
    return clean(field.replace("_comma_", ","))


def parse_ed_csv(text: str) -> dict[str, list[EdUtterance]]:
    """The released CSVs escape commas inside fields as `_comma_`, so a line
    splits on commas (as the authors' loader does); valid/test lines carry
    extra columns after `tags`, which are ignored."""
    conversations: dict[str, list[EdUtterance]] = defaultdict(list)
    lines = text.split("\n")
    for line in lines[1:]:
        parts = line.rstrip("\r").split(",")
        if len(parts) < 6:
            continue
        conv_id, index, emotion, situation, _speaker, utterance = parts[:6]
        conversations[conv_id].append(
            EdUtterance(conv_id, int(index), emotion, _ed_text(situation), _ed_text(utterance))
        )
    for utterances in conversations.values():
        utterances.sort(key=lambda u: u.index)
    return dict(conversations)


def ed_blocks(utterances: Sequence[EdUtterance]) -> list[Block]:
    """The first speaker (odd utterance_idx) is the person who had the
    experience; the second is the listener."""
    return join_runs(("user" if u.index % 2 else "assistant", u.text, None) for u in utterances)


def ed_items(
    splits: Mapping[Split, Mapping[str, Sequence[EdUtterance]]], per_split: int, seed: int
) -> list[DialogueItem]:
    """Official splits (valid -> dev, test -> test); conversations taken in a
    `seed`-shuffled order, one cut each."""
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    items: list[DialogueItem] = []
    for split, conversations in splits.items():
        ids = sorted(conversations)
        rng.shuffle(ids)
        taken = 0
        for conv_id in ids:
            if taken == per_split:
                break
            utterances = conversations[conv_id]
            blocks = ed_blocks(utterances)
            cuts = cut_points(blocks)
            if not cuts:
                continue
            cut = rng.choice(cuts)
            slug = re.sub(r"[^A-Za-z0-9]+", "-", conv_id).strip("-")
            items.append(
                _item(
                    f"ed-{slug}-{cut:02d}",
                    "empatheticdialogues",
                    split,
                    blocks,
                    cut,
                    {
                        "conversation": conv_id,
                        "emotion": utterances[0].emotion,
                        "situation": utterances[0].situation,
                    },
                )
            )
            taken += 1
    return items


# --- measures against the human reference -------------------------------
# Proxies, not quality: they say whether the companion took the same broad
# move as the human at that point (asked, suggested) and how long it ran.

_SUGGESTION_STRATEGIES = frozenset({"Providing Suggestions"})
_QUESTION_STRATEGIES = frozenset({"Question"})


def human_moves(reference: Reference) -> dict[str, bool]:
    strategies = set(reference.strategies)
    return {
        "asks": "?" in reference.reply or bool(strategies & _QUESTION_STRATEGIES),
        "suggests": bool(strategies & _SUGGESTION_STRATEGIES)
        if strategies
        else bool(ADVICE_WORDING.search(reference.reply)),
    }


def compare(reply: str, item: DialogueItem) -> dict[str, Any]:
    human = human_moves(item.reference)
    asks = "?" in reply
    suggests = bool(ADVICE_WORDING.search(reply))
    return {
        "reply_words": words(reply),
        "reference_words": words(item.reference.reply),
        "length_ratio": round(words(reply) / max(words(item.reference.reply), 1), 2),
        "asks": asks,
        "human_asks": human["asks"],
        "suggests": suggests,
        "human_suggests": human["suggests"],
    }


def summarize_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"items": 0}

    def rate(predicate: Any) -> float | None:
        selected = [r for r in rows if predicate(r)]
        return round(len(selected) / n, 3)

    def conditional(given: str, then: str, value: bool) -> float | None:
        selected = [r for r in rows if r[given] is value]
        if not selected:
            return None
        return round(sum(1 for r in selected if r[then]) / len(selected), 3)

    ratios = sorted(r["length_ratio"] for r in rows)
    return {
        "items": n,
        "median_length_ratio": ratios[n // 2]
        if n % 2
        else (ratios[n // 2 - 1] + ratios[n // 2]) / 2,
        "asks_rate": rate(lambda r: r["asks"]),
        "human_asks_rate": rate(lambda r: r["human_asks"]),
        "asks_when_human_asks": conditional("human_asks", "asks", True),
        "asks_when_human_does_not": conditional("human_asks", "asks", False),
        "suggests_rate": rate(lambda r: r["suggests"]),
        "human_suggests_rate": rate(lambda r: r["human_suggests"]),
        "suggests_when_human_does_not": conditional("human_suggests", "suggests", False),
    }

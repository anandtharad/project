"""Training-data preparation (spec §17.3, Phase 7). No training run.

    source files -> licence gate -> conversations -> normalisation
      -> examples -> quality filter -> safety screen -> strategy labels
      -> deduplication -> split -> leakage check -> canonical records

Every stage is deterministic, so the same source files, evaluation sets and
settings give byte-identical records and the same build id; the manifest
records what each stage dropped and why. A source is read only when
training/sources.toml approves it.

Examples: each supporter/listener message that answers a person's message
is a target; the history is the conversation before it (at most
`MAX_HISTORY` messages, starting with the person), as the companion sees
history at run time.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.dialogue import Block, ed_blocks, esconv_blocks, parse_ed_csv
from app.evaluation.policy_effect import ADVICE_WORDING
from app.orchestration.policy import explicit_request
from app.rag.extract import normalize_text
from app.safety.patterns import PATTERNS_VERSION, screen
from app.training.leakage import LeakageIndex
from app.training.schema import (
    OPTIONAL_FIELDS,
    Message,
    Provenance,
    QualityLabels,
    SafetyLabels,
    Split,
    Strategy,
    SystemRef,
    TrainingRecord,
)
from app.training.sources import Registry, Source

PIPELINE_VERSION = "1.0.0"
MAX_HISTORY = 8
MIN_MESSAGE_WORDS = 2
MIN_TARGET_WORDS = 3
MAX_TARGET_WORDS = 200
# The same reply word for word ("That sounds hard.") at most this many times.
MAX_SAME_TARGET = 3
VALIDATION_SHARE = 10  # one conversation in ten

# ESConv supporter strategies -> the companion's response strategies
# (app/orchestration/policy.py). Self-disclosure and Information are not
# mapped: they are dropped as targets (see `quality_reason`).
ESCONV_STRATEGIES = {
    "Question": "explore",
    "Restatement or Paraphrasing": "reflect",
    "Reflection of feelings": "reflect",
    "Affirmation and Reassurance": "validate",
    "Providing Suggestions": "problem_solve",
}

_WORD = re.compile(r"[A-Za-z0-9']+")
_CONTACT = re.compile(r"https?://|www\.|\b[\w.+-]+@[\w-]+\.\w|\b\d[\d ()-]{6,}\d\b", re.IGNORECASE)
# The companion never claims a life of its own (companion_system: "Don't
# claim human experiences, a body, a personal history, or feelings of your
# own"), so a human supporter's own experience is not a target to learn.
_SELF_DISCLOSURE = re.compile(
    r"\b(?:i (?:had|went through|used to|lost|once|experienced|remember when)"
    r"|i was (?!wondering|going to|just|thinking|hoping|asking)\w+"
    r"|i've (?:had|been through|lost|gone through|experienced)"
    r"|me too\b|same here|happened to me|in my (?:experience|case|life)"
    r"|my (?:wife|husband|son|daughter|kids?|children|mom|mum|dad|mother|father|brother|sister|"
    r"boss|friends?|dog|cat|family|girlfriend|boyfriend|partner|grandma|grandmother|grandpa|"
    r"grandfather|job|cousin|aunt|uncle))\b",
    re.IGNORECASE,
)
# Traces of the crowd-work setting, not of a conversation.
_CROWDWORK = re.compile(
    r"\b(?:mturk|turkers?|survey|bonus|this (?:task|study|chat session)|the task)\b",
    re.IGNORECASE,
)


def words(text: str) -> int:
    return len(_WORD.findall(text))


def clean(text: str) -> str:
    return normalize_text(text)


@dataclass(frozen=True)
class Conversation:
    source: str
    conversation_id: str
    blocks: tuple[Block, ...]
    labels: dict[str, Any]


@dataclass(frozen=True)
class Example:
    conversation: Conversation
    turn: int
    history: tuple[Message, ...]
    history_truncated: bool
    message: str
    target: str
    source_strategies: tuple[str, ...]


@dataclass
class StageCounts:
    conversations: Counter[str] = field(default_factory=Counter)
    examples: Counter[str] = field(default_factory=Counter)
    dropped: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def drop(self, source: str, reason: str) -> None:
        self.dropped[source][reason] += 1


# --- source adapters ------------------------------------------------------


def esconv_conversations(data: bytes) -> Iterator[Conversation]:
    for index, raw in enumerate(json.loads(data.decode("utf-8"))):
        survey = raw.get("survey_score") or {}
        yield Conversation(
            "esconv",
            str(index),
            tuple(esconv_blocks(raw)),
            {
                "problem": str(raw.get("problem_type", "")).lower(),
                "emotion": raw.get("emotion_type"),
                "seeker_survey": survey.get("seeker", {}),
            },
        )


def ed_train_conversations(archive: bytes) -> Iterator[Conversation]:
    """The official train split only; valid and test are evaluation data."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember("empatheticdialogues/train.csv")
        handle = tar.extractfile(member)
        if handle is None:
            raise ValueError("empatheticdialogues/train.csv is not a file")
        text = handle.read().decode("utf-8")
    for conv_id, utterances in sorted(parse_ed_csv(text).items()):
        yield Conversation(
            "empatheticdialogues",
            conv_id,
            tuple(ed_blocks(utterances)),
            {"emotion": utterances[0].emotion},
        )


ADAPTERS = {"esconv": esconv_conversations, "empatheticdialogues": ed_train_conversations}


# --- stages ---------------------------------------------------------------


def examples(conversation: Conversation) -> Iterator[Example]:
    blocks = [Block(b.role, clean(b.text), b.strategies) for b in conversation.blocks]
    for i in range(1, len(blocks)):
        if blocks[i].role != "assistant" or blocks[i - 1].role != "user":
            continue
        cut = i - 1
        start = max(0, cut - MAX_HISTORY)
        while start < cut and blocks[start].role != "user":
            start += 1
        yield Example(
            conversation,
            i,
            tuple(Message(role=b.role, content=b.text) for b in blocks[start:cut]),
            start > 0,
            blocks[cut].text,
            blocks[i].text,
            blocks[i].strategies,
        )


def quality_reason(example: Example) -> str | None:
    """Why an example is not a good target, or None."""
    if words(example.message) < MIN_MESSAGE_WORDS:
        return "message_too_short"
    target_words = words(example.target)
    if target_words < MIN_TARGET_WORDS:
        return "target_too_short"
    if target_words > MAX_TARGET_WORDS:
        return "target_too_long"
    if _CONTACT.search(example.target):
        return "contact_or_link"
    if "Self-disclosure" in example.source_strategies or _SELF_DISCLOSURE.search(example.target):
        return "self_disclosure"
    if "Information" in example.source_strategies:
        # Facts without evidence would teach unsupported claims (spec §11.9).
        return "unsupported_information"
    if _CROWDWORK.search(example.target) or _CROWDWORK.search(example.message):
        return "crowdwork_artifact"
    if not explicit_request(example.message)[1] and (
        "Providing Suggestions" in example.source_strategies
        or ADVICE_WORDING.search(example.target)
    ):
        # The person said they don't want advice; the companion honours that.
        return "advice_after_no_advice"
    previous = [m.content for m in example.history if m.role == "assistant"]
    if previous and previous[-1].lower() == example.target.lower():
        return "repeats_previous"
    return None


def safety_hits(example: Example) -> tuple[list[str], list[str]]:
    """Rules of the deterministic screen firing on the message (with the
    person's previous message for imminence cues) and on the target."""
    previous = next((m.content for m in reversed(example.history) if m.role == "user"), None)
    return (
        list(screen(example.message, previous).rule_ids),
        list(screen(example.target).rule_ids),
    )


def strategy_for(example: Example) -> Strategy | None:
    mapped = [ESCONV_STRATEGIES[s] for s in example.source_strategies if s in ESCONV_STRATEGIES]
    if not mapped:
        return None
    return Strategy(
        value=mapped[0], annotation="source_label", source_labels=list(example.source_strategies)
    )


def split_for(source: str, conversation_id: str) -> Split:
    """By conversation, so no conversation is in both splits."""
    digest = hashlib.sha256(f"{source}:{conversation_id}".encode()).digest()
    return "validation" if digest[0] % VALIDATION_SHARE == 0 else "train"


def _normal(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


# --- build ----------------------------------------------------------------


@dataclass(frozen=True)
class SourceInput:
    source: Source
    data: bytes


@dataclass
class Build:
    records: list[TrainingRecord]
    manifest: dict[str, Any]


def build(
    inputs: Sequence[SourceInput],
    registry: Registry,
    leakage: LeakageIndex,
    system: SystemRef,
    adapters: dict[str, Any] | None = None,
) -> Build:
    adapters = adapters or ADAPTERS
    counts = StageCounts()
    kept: list[tuple[Example, Strategy | None, list[str], list[str]]] = []
    seen_examples: set[str] = set()
    target_uses: Counter[str] = Counter()
    for item in inputs:
        source = registry.get(item.source.id)
        if not source.approval.approved:
            raise PermissionError(
                f"{source.id} is not approved for training (training/sources.toml)"
            )
        digest = hashlib.sha256(item.data).hexdigest()
        if digest != source.sha256:
            raise ValueError(f"{source.id}: file sha256 {digest} is not the audited file")
        for conversation in adapters[source.id](item.data):
            counts.conversations[source.id] += 1
            if (source.id, conversation.conversation_id) in leakage.conversations:
                counts.drop(source.id, "evaluation_conversation")
                continue
            for example in examples(conversation):
                counts.examples[source.id] += 1
                message_hits: list[str] = []
                target_hits: list[str] = []
                reason = quality_reason(example)
                if reason is None:
                    message_hits, target_hits = safety_hits(example)
                    if message_hits or target_hits:
                        reason = "safety_screen"
                if reason is None and any(
                    leakage.leaks(t)
                    for t in (
                        example.message,
                        example.target,
                        *(m.content for m in example.history),
                    )
                ):
                    reason = "evaluation_text"
                if reason is None:
                    key = hashlib.sha256(
                        f"{_normal(example.message)}\n{_normal(example.target)}".encode()
                    ).hexdigest()
                    if key in seen_examples:
                        reason = "duplicate_example"
                    elif target_uses[_normal(example.target)] >= MAX_SAME_TARGET:
                        reason = "frequent_target"
                    else:
                        seen_examples.add(key)
                        target_uses[_normal(example.target)] += 1
                if reason is not None:
                    counts.drop(source.id, reason)
                    continue
                kept.append((example, strategy_for(example), message_hits, target_hits))

    records = [
        _record(
            example,
            strategy,
            message_hits,
            target_hits,
            registry.get(example.conversation.source),
            system,
        )
        for example, strategy, message_hits, target_hits in kept
    ]
    records.sort(key=lambda r: r.id)
    by_split = Counter((r.provenance.source, r.split) for r in records)
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": records[0].schema_version if records else None,
        "system": system.model_dump(),
        "sources": {
            item.source.id: {
                "file_sha256": hashlib.sha256(item.data).hexdigest(),
                "licence": item.source.licence,
                "approved_by": item.source.approval.approved_by,
                "approved_on": item.source.approval.approved_on,
                "conversations": counts.conversations[item.source.id],
                "examples": counts.examples[item.source.id],
                "dropped": dict(sorted(counts.dropped[item.source.id].items())),
                "records": {
                    split: by_split[(item.source.id, split)] for split in ("train", "validation")
                },
            }
            for item in inputs
        },
        "settings": {
            "max_history": MAX_HISTORY,
            "min_message_words": MIN_MESSAGE_WORDS,
            "min_target_words": MIN_TARGET_WORDS,
            "max_target_words": MAX_TARGET_WORDS,
            "max_same_target": MAX_SAME_TARGET,
            "validation_share": f"1/{VALIDATION_SHARE} of conversations",
            "screen_version": PATTERNS_VERSION,
        },
        "evaluation_index": {
            "files": leakage.files,
            "texts": len(leakage.texts),
            "conversations": len(leakage.conversations),
        },
        "records": len(records),
        "records_sha256": records_sha256(records),
    }
    manifest["build_id"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items()}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return Build(records, manifest)


def _record(
    example: Example,
    strategy: Strategy | None,
    message_hits: list[str],
    target_hits: list[str],
    source: Source,
    system: SystemRef,
) -> TrainingRecord:
    conversation = example.conversation
    values: dict[str, Any] = {
        "state": None,
        "strategy": strategy,
        "memory": None,
        "knowledge": None,
    }
    return TrainingRecord(
        id=f"{conversation.source}-{_slug(conversation.conversation_id)}-{example.turn:03d}",
        split=split_for(conversation.source, conversation.conversation_id),
        system=system,
        history=list(example.history),
        message=example.message,
        target=example.target,
        absent=[f for f in OPTIONAL_FIELDS if values[f] is None],
        provenance=Provenance(
            source=source.id,
            source_url=source.url,
            licence=source.licence,
            file_sha256=source.sha256,
            conversation=conversation.conversation_id,
            turn=example.turn,
            synthetic=source.synthetic,
            transformations=[
                f"join_same_speaker_runs@{PIPELINE_VERSION}",
                f"normalize_text@{PIPELINE_VERSION}",
                f"history_last_{MAX_HISTORY}@{PIPELINE_VERSION}",
                f"quality_filter@{PIPELINE_VERSION}",
                f"safety_screen@{PATTERNS_VERSION}",
                f"leakage_check@{PIPELINE_VERSION}",
                f"dedup@{PIPELINE_VERSION}",
                *([f"strategy_from_source_label@{PIPELINE_VERSION}"] if strategy else []),
            ],
            pipeline_version=PIPELINE_VERSION,
        ),
        quality=QualityLabels(
            target_words=words(example.target),
            message_words=words(example.message),
            history_messages=len(example.history),
            history_truncated=example.history_truncated,
            source=conversation.labels,
        ),
        safety=SafetyLabels(
            screen_version=PATTERNS_VERSION, message_hits=message_hits, target_hits=target_hits
        ),
        **values,
    )


def records_jsonl(records: Iterable[TrainingRecord]) -> str:
    return "".join(
        json.dumps(r.model_dump(), ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )


def records_sha256(records: Iterable[TrainingRecord]) -> str:
    return hashlib.sha256(records_jsonl(records).encode()).hexdigest()

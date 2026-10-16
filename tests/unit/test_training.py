"""Training-data preparation (app/training/): the licence gate, every
filtering stage, leakage checks, splits, provenance and reproducibility, on
a synthetic fixture source."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evaluation.dialogue import Block
from app.training import pipeline
from app.training.leakage import LeakageIndex, build_index
from app.training.pipeline import Conversation, SourceInput, build
from app.training.schema import Message, SystemRef, TrainingRecord
from app.training.sources import Approval, Registry, Source, load_registry

SYSTEM = SystemRef(prompt="companion_system@1.7.0", checksum="x")
REPLY = "that sounds really draining, what has the week been like"


def fixture_conversations(data: bytes) -> Iterator[Conversation]:
    for raw in json.loads(data):
        yield Conversation(
            "fixture",
            raw["id"],
            tuple(Block(role, text, tuple(tags)) for role, text, tags in raw["turns"]),
            {"emotion": "tired"},
        )


def source(data: bytes, approved: bool = True) -> Source:
    return Source(
        id="fixture",
        name="fixture",
        url="https://example.invalid/fixture",
        file="fixture.json",
        file_url="https://example.invalid/fixture.json",
        sha256=hashlib.sha256(data).hexdigest(),
        licence="CC0-1.0",
        permitted_use="tests",
        citation="-",
        synthetic=True,
        origin="written for the tests",
        splits="by conversation",
        quality_annotations="none",
        licence_checked_on="2026-10-16",
        approval=Approval(
            approved=approved,
            approved_by="tests" if approved else "",
            approved_on="2026-10-16" if approved else "",
        ),
    )


def run(conversations: list[dict[str, Any]], leakage: LeakageIndex | None = None) -> Any:
    data = json.dumps(conversations).encode()
    src = source(data)
    return build(
        [SourceInput(src, data)],
        Registry(sources_version=1, sources=[src]),
        leakage or LeakageIndex(),
        SYSTEM,
        adapters={"fixture": fixture_conversations},
    )


def conversation(
    cid: str, reply: str = REPLY, tags: tuple[str, ...] = ("Question",)
) -> dict[str, Any]:
    return {
        "id": cid,
        "turns": [
            ["assistant", "hi there, how can i help", []],
            ["user", "work has been a lot this week honestly", []],
            ["assistant", reply, list(tags)],
        ],
    }


def test_unapproved_or_altered_sources_are_refused() -> None:
    data = json.dumps([conversation("c1")]).encode()
    unapproved = source(data, approved=False)
    with pytest.raises(PermissionError):
        build(
            [SourceInput(unapproved, data)],
            Registry(sources_version=1, sources=[unapproved]),
            LeakageIndex(),
            SYSTEM,
            adapters={"fixture": fixture_conversations},
        )
    approved = source(data)
    with pytest.raises(ValueError, match="audited file"):
        build(
            [SourceInput(approved, data + b" ")],
            Registry(sources_version=1, sources=[approved]),
            LeakageIndex(),
            SYSTEM,
            adapters={"fixture": fixture_conversations},
        )
    with pytest.raises(ValidationError):
        Approval(approved=True)  # an approval names who and when


def test_records_have_provenance_strategy_and_explicit_absence() -> None:
    result = run([conversation("c1")])
    [record] = result.records
    assert record.message == "work has been a lot this week honestly"
    assert record.target == REPLY
    assert record.history == []  # the opening greeting is before the person's message
    assert record.strategy is not None and record.strategy.value == "explore"
    assert record.strategy.source_labels == ["Question"]
    assert record.absent == ["state", "memory", "knowledge"]
    assert record.provenance.source == "fixture" and record.provenance.conversation == "c1"
    assert record.provenance.turn == 2 and record.provenance.synthetic is True
    assert record.safety.message_hits == [] and record.safety.target_hits == []
    manifest = result.manifest["sources"]["fixture"]
    assert manifest["conversations"] == 1 and manifest["examples"] == 1
    assert sum(manifest["records"].values()) == 1


@pytest.mark.parametrize(
    ("reply", "tags", "reason"),
    [
        ("ok", (), "target_too_short"),
        ("call them on 0800 123 4567 tonight if you can", (), "contact_or_link"),
        ("have a look at https://example.com it really helps", (), "contact_or_link"),
        ("i went through the same thing last year and it got better", (), "self_disclosure"),
        ("that sounds hard, honestly it does", ("Self-disclosure",), "self_disclosure"),
        (
            "caffeine stays in your body for about six hours",
            ("Information",),
            "unsupported_information",
        ),
        ("thanks for doing this task with me today friend", (), "crowdwork_artifact"),
    ],
)
def test_quality_filter_drops_with_a_reason(reply: str, tags: tuple[str, ...], reason: str) -> None:
    result = run([conversation("c1", reply, tags)])
    assert result.records == []
    assert result.manifest["sources"]["fixture"]["dropped"] == {reason: 1}


def test_safety_screen_hits_drop_the_example() -> None:
    risky = {
        "id": "c1",
        "turns": [
            ["user", "i want to kill myself tonight", []],
            ["assistant", "i am so glad you told me, are you safe right now", ["Question"]],
        ],
    }
    result = run([risky])
    assert result.records == []
    assert result.manifest["sources"]["fixture"]["dropped"] == {"safety_screen": 1}


def test_evaluation_conversations_and_texts_are_never_used() -> None:
    leakage = LeakageIndex()
    leakage.conversations.add(("fixture", "c1"))
    leakage.add_text("work has been a lot this week honestly")  # c2's message, verbatim
    result = run(
        [conversation("c1"), conversation("c2", "what part of the week was worst for you")], leakage
    )
    assert result.records == []
    assert result.manifest["sources"]["fixture"]["dropped"] == {
        "evaluation_conversation": 1,
        "evaluation_text": 1,
    }
    # An 8-word run shared with an evaluation text is a leak too.
    leakage = LeakageIndex()
    leakage.add_text("i keep thinking that sounds really draining what has the week been like")
    assert leakage.leaks(REPLY)
    assert not leakage.leaks("a completely different reply about the weekend plans")


def test_duplicates_and_frequent_targets_are_capped() -> None:
    same = [conversation(f"c{i}") for i in range(3)]  # identical message and target
    result = run(same)
    assert len(result.records) == 1
    assert result.manifest["sources"]["fixture"]["dropped"] == {"duplicate_example": 2}
    varied = [
        {
            "id": f"v{i}",
            "turns": [["user", f"message number {i} about my week", []], ["assistant", REPLY, []]],
        }
        for i in range(5)
    ]
    result = run(varied)
    assert len(result.records) == pipeline.MAX_SAME_TARGET
    assert result.manifest["sources"]["fixture"]["dropped"] == {"frequent_target": 2}


def test_history_is_bounded_and_starts_with_the_person() -> None:
    turns = []
    for i in range(12):
        turns.append(["user", f"person message number {i} here", []])
        turns.append(["assistant", f"companion reply number {i} for you", []])
    result = run([{"id": "long", "turns": turns}])
    last = max(result.records, key=lambda r: r.provenance.turn)
    assert len(last.history) <= pipeline.MAX_HISTORY
    assert last.history[0].role == "user" and last.quality.history_truncated


def test_splits_are_by_conversation_and_builds_are_reproducible() -> None:
    conversations = []
    for i in range(60):
        turns = [
            ["user", f"first thing on my mind number {i}", []],
            ["assistant", f"tell me more about thing {i} please", []],
            ["user", f"second thing on my mind number {i}", []],
            ["assistant", f"and how did thing {i} make you feel", []],
        ]
        conversations.append({"id": f"s{i}", "turns": turns})
    first, second = run(conversations), run(conversations)
    assert first.manifest == second.manifest
    assert pipeline.records_jsonl(first.records) == pipeline.records_jsonl(second.records)
    splits: dict[str, set[str]] = {}
    for record in first.records:
        splits.setdefault(record.provenance.conversation, set()).add(record.split)
    assert all(len(s) == 1 for s in splits.values())
    assert {r.split for r in first.records} == {"train", "validation"}


def test_record_schema_requires_exact_absence() -> None:
    record = run([conversation("c1")]).records[0]
    data = record.model_dump()
    with pytest.raises(ValidationError):
        TrainingRecord.model_validate({**data, "absent": ["state"]})
    with pytest.raises(ValidationError):
        TrainingRecord.model_validate(
            {**data, "history": [Message(role="assistant", content="hi").model_dump()]}
        )


def test_registry_has_no_approved_source_by_default() -> None:
    registry = load_registry()
    assert {s.id for s in registry.sources} == {"esconv", "empatheticdialogues"}
    assert not any(s.approval.approved for s in registry.sources)


def test_leakage_index_reads_every_evaluation_file(tmp_path: Path) -> None:
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "a.jsonl").write_text(
        json.dumps({"id": "x", "steps": [{"user": "a long enough evaluation message here"}]})
        + "\n"
        + json.dumps({"source": "esconv", "meta": {"conversation": 7}, "turns": ["hi"]})
        + "\n"
    )
    voice = tmp_path / "voice.json"
    voice.write_text(
        json.dumps(
            {"sets": {"held_out_public": {"judged": {"esconv-12": {}, "ed-hit:1_conv:2": {}}}}}
        )
    )
    index = build_index(tmp_path, voice)
    assert index.leaks("A long enough evaluation message here!")
    assert index.conversations == {
        ("esconv", "7"),
        ("esconv", "12"),
        ("empatheticdialogues", "hit:1_conv:2"),
    }
    assert list(index.files) == ["set/a.jsonl"]


def test_esconv_adapter_joins_runs_and_keeps_labels() -> None:
    data = json.dumps(
        [
            {
                "problem_type": "Job Crisis",
                "emotion_type": "anxiety",
                "survey_score": {"seeker": {"empathy": "5"}},
                "dialog": [
                    {"speaker": "seeker", "annotation": {}, "content": "hi\n"},
                    {
                        "speaker": "seeker",
                        "annotation": {},
                        "content": "my job is stressing me out",
                    },
                    {
                        "speaker": "supporter",
                        "annotation": {"strategy": "Question"},
                        "content": "what is the hardest part?",
                    },
                ],
            }
        ]
    ).encode()
    [conversation] = list(pipeline.esconv_conversations(data))
    assert conversation.conversation_id == "0"
    assert [b.role for b in conversation.blocks] == ["user", "assistant"]
    assert conversation.blocks[0].text == "hi my job is stressing me out"
    assert conversation.blocks[1].strategies == ("Question",)
    assert conversation.labels["problem"] == "job crisis"


def test_ed_adapter_reads_only_the_train_split(tmp_path: Path) -> None:
    import io
    import tarfile

    header = "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags\n"
    files = {
        "train.csv": header
        + "hit:1_conv:2,1,sad,My dog_comma_ Rex,1,I lost my dog last week_comma_ still sad,,\n"
        + "hit:1_conv:2,2,sad,My dog_comma_ Rex,2,I am so sorry_comma_ that is hard,,\n",
        "valid.csv": header + "hit:9_conv:9,1,joy,x,1,this must never be read,,\n",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            raw = text.encode()
            info = tarfile.TarInfo(f"empatheticdialogues/{name}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    conversations = list(pipeline.ed_train_conversations(buffer.getvalue()))
    assert [c.conversation_id for c in conversations] == ["hit:1_conv:2"]
    assert conversations[0].blocks[0].text == "I lost my dog last week, still sad"
    assert conversations[0].blocks[1].role == "assistant"


def test_advice_after_the_person_declined_it_is_dropped() -> None:
    declined = {
        "id": "c1",
        "turns": [
            ["user", "i don't want advice, i just need to vent about my boss", []],
            ["assistant", "you could try talking to hr about it", ["Providing Suggestions"]],
        ],
    }
    result = run([declined])
    assert result.manifest["sources"]["fixture"]["dropped"] == {"advice_after_no_advice": 1}

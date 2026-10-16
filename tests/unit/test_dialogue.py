"""Dialogue benchmark items (app/evaluation/dialogue.py) and the runner's
packets and scoring (scripts/eval_dialogue.py), on synthetic data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from app.evaluation import dialogue
from app.evaluation.dialogue import (
    Block,
    DialogueItem,
    Reference,
    compare,
    context_before,
    cut_points,
    ed_items,
    esconv_items,
    join_runs,
    parse_ed_csv,
    summarize_comparisons,
)

ROOT = Path(__file__).resolve().parents[2]
LONG = "this is a message long enough to answer"


def load_runner():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "eval_dialogue", ROOT / "scripts" / "eval_dialogue.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_dialogue"] = module
    spec.loader.exec_module(module)
    return module


def esconv_conversation(n_exchanges: int = 3) -> dict[str, Any]:
    dialog: list[dict[str, Any]] = [
        {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "Hi there"}
    ]
    for i in range(n_exchanges):
        dialog.append({"speaker": "seeker", "annotation": {}, "content": f"{LONG} {i}\n"})
        dialog.append({"speaker": "seeker", "annotation": {}, "content": "and more"})
        dialog.append(
            {
                "speaker": "supporter",
                "annotation": {"strategy": "Reflection of feelings"},
                "content": f"that sounds hard to carry {i}",
            }
        )
        dialog.append(
            {
                "speaker": "supporter",
                "annotation": {"strategy": "Question"},
                "content": "what happened next?",
            }
        )
    return {
        "problem_type": "Job Crisis",
        "emotion_type": "anxiety",
        "experience_type": "Current Experience",
        "situation": "  worried\nabout work ",
        "dialog": dialog,
    }


def test_join_runs_merges_same_speaker_and_keeps_strategies() -> None:
    blocks = join_runs(
        [
            ("user", "one ", None),
            ("user", " two", None),
            ("assistant", "a", "Question"),
            ("assistant", "", "Others"),  # empty text dropped
            ("assistant", "b", "Information"),
        ]
    )
    assert blocks == [
        Block("user", "one two"),
        Block("assistant", "a b", ("Question", "Information")),
    ]


def test_cut_points_need_a_long_enough_message_and_reply() -> None:
    blocks = [
        Block("user", "hi"),
        Block("assistant", "hello how are you today"),
        Block("user", LONG),
        Block("assistant", "ok"),
        Block("user", LONG),
        Block("assistant", "that sounds hard to carry"),
    ]
    assert cut_points(blocks) == [4]


def test_context_is_bounded_and_starts_with_the_person(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dialogue, "MAX_CONTEXT_MESSAGES", 3)
    blocks = [Block("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(8)]
    context, truncated = context_before(blocks, 6)
    # The window would start at blocks[3] (assistant); it moves to blocks[4].
    assert [m.content for m in context] == ["m4", "m5"]
    assert truncated
    context, truncated = context_before(blocks, 2)
    assert [m.content for m in context] == ["m0", "m1"] and not truncated


def test_esconv_items_are_seeded_split_and_well_formed() -> None:
    conversations = [esconv_conversation() for _ in range(10)]
    conversations[3] = {**esconv_conversation(), "dialog": []}  # no cut point: skipped
    first = esconv_items(conversations, per_split=2, seed=1)
    assert [i.model_dump() for i in first] == [
        i.model_dump() for i in esconv_items(conversations, per_split=2, seed=1)
    ]
    assert [i.split for i in first] == ["dev", "dev", "test", "test"]
    assert len({i.meta["conversation"] for i in first}) == 4
    item = first[0]
    assert item.source == "esconv"
    assert item.message.startswith(LONG) and item.message.endswith("and more")
    assert item.reference.strategies == ["Reflection of feelings", "Question"]
    assert item.meta["problem"] == "job crisis"
    assert item.meta["situation"] == "worried about work"
    # The context ends with the supporter's reply before the cut, and starts
    # with the seeker (the opening "Hi there" is dropped).
    assert item.context == [] or item.context[0].role == "user"


ED_CSV = (
    "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags\n"
    "hit:1_conv:2,1,sad,My dog_comma_ Rex_comma_ died,10,"
    "My dog died last week_comma_ I still miss him a lot,5|5|5_5|5|5,\n"
    "hit:1_conv:2,2,sad,My dog_comma_ Rex_comma_ died,11,"
    "Oh no_comma_ I am so sorry to hear that,5|5|5_5|5|5,,extra|candidate\n"
    "hit:1_conv:2,3,sad,My dog_comma_ Rex_comma_ died,10,thanks,5|5|5_5|5|5,\n"
    "hit:3_conv:4,1,proud,Won a race,12,I finally won the town race yesterday,,\n"
)


def test_parse_ed_csv_unescapes_commas_and_ignores_extra_columns() -> None:
    conversations = parse_ed_csv(ED_CSV)
    assert sorted(conversations) == ["hit:1_conv:2", "hit:3_conv:4"]
    dog = conversations["hit:1_conv:2"]
    assert [u.index for u in dog] == [1, 2, 3]
    assert dog[0].situation == "My dog, Rex, died"
    assert dog[1].text == "Oh no, I am so sorry to hear that"


def test_ed_items_use_official_splits_and_skip_conversations_without_a_reply() -> None:
    conversations = parse_ed_csv(ED_CSV)
    items = ed_items({"dev": conversations, "test": {}}, per_split=5, seed=3)
    # hit:3_conv:4 has no listener reply, so only the dog conversation counts.
    assert len(items) == 1
    item = items[0]
    assert item.split == "dev" and item.source == "empatheticdialogues"
    assert item.id == "ed-hit-1-conv-2-00"
    assert item.context == []
    assert item.message == "My dog died last week, I still miss him a lot"
    assert item.reference == Reference(reply="Oh no, I am so sorry to hear that")
    assert item.meta["emotion"] == "sad"


def make_item(item_id: str, reply: str, strategies: list[str] | None = None) -> DialogueItem:
    return DialogueItem(
        id=item_id,
        source="esconv",
        split="dev",
        context=[],
        turns=[LONG],
        reference=Reference(reply=reply, strategies=strategies or []),
    )


def test_compare_reads_moves_from_strategies_or_wording() -> None:
    labelled = make_item("a", "You might try a walk.", ["Providing Suggestions"])
    row = compare("What happened after that?", labelled)
    assert row["asks"] and not row["suggests"]
    assert row["human_suggests"] and not row["human_asks"]
    # Without strategy labels (EmpatheticDialogues), wording decides.
    unlabelled = make_item("b", "Maybe you could try calling her?")
    row = compare("You could try writing it down.", unlabelled)
    assert row["human_asks"] and row["human_suggests"] and row["suggests"]
    assert row["length_ratio"] == round(6 / 6, 2)


def test_summarize_comparisons_conditions_on_the_human_move() -> None:
    rows = [
        compare("What then?", make_item("1", "And then what?")),
        compare("That sounds heavy.", make_item("2", "What did you do?")),
        compare("Try resting.", make_item("3", "I am sorry.")),
    ]
    summary = summarize_comparisons(rows)
    assert summary["items"] == 3
    assert summary["asks_when_human_asks"] == 0.5
    assert summary["asks_when_human_does_not"] == 0.0
    assert summary["suggests_when_human_does_not"] == round(1 / 3, 3)
    assert summarize_comparisons([]) == {"items": 0}


def test_packets_are_blind_and_the_key_recovers_the_companion() -> None:
    runner = load_runner()
    items = [make_item(f"i{n}", f"human {n}") for n in range(20)]
    replies = {i.id: f"companion {i.id}" for i in items}
    blind, key = runner.packets(items, replies, seed=5)
    assert blind == runner.packets(items, replies, seed=5)[0]
    for packet in blind:
        letter = next(k for k, v in key[packet["id"]].items() if v == "companion")
        assert packet["replies"][letter] == replies[packet["id"]]
        assert set(packet) == {"id", "context", "message", "replies"}
    # Both orders occur.
    assert {key[i.id]["A"] for i in items} == {"companion", "human"}


def test_pick_alternates_sources_within_the_split() -> None:
    runner = load_runner()
    items = [
        make_item("e1", "x"),
        make_item("e2", "x"),
        make_item("e3", "x").model_copy(update={"split": "test"}),
        make_item("d1", "x").model_copy(update={"source": "empatheticdialogues"}),
    ]
    assert [i.id for i in runner.pick(items, "dev", None)] == ["e1", "d1", "e2"]
    assert [i.id for i in runner.pick(items, "dev", 2)] == ["e1", "d1"]


def test_score_maps_letters_to_outcomes_and_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "JUDGE_DIR", tmp_path)
    result = tmp_path / "dialogue-dev-x.json"
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {"id": "a", "source": "esconv"},
                    {"id": "b", "source": "esconv"},
                    {"id": "c", "source": "empatheticdialogues"},
                ]
            }
        )
    )
    (tmp_path / "dialogue-dev-x.key.json").write_text(
        json.dumps(
            {
                "a": {"A": "companion", "B": "human"},
                "b": {"A": "human", "B": "companion"},
                "c": {"A": "companion", "B": "human"},
            }
        )
    )
    j1 = tmp_path / "j1.json"
    j2 = tmp_path / "j2.json"
    j1.write_text(
        json.dumps(
            [
                {"id": "a", "preference": "A"},  # companion wins
                {"id": "b", "preference": "A"},  # human wins
                {"id": "c", "preference": "tie"},
            ]
        )
    )
    j2.write_text(
        json.dumps(
            [
                {"id": "a", "preference": "A"},
                {"id": "b", "preference": "B"},  # disagrees with j1
                {"id": "c", "preference": "tie"},
            ]
        )
    )
    args = type("Args", (), {"result": str(result), "judgements": [str(j1), str(j2)]})()
    assert runner.score(args) == 0
    judged = json.loads(result.read_text())["judgement"]
    assert judged["judges"]["j1"]["all"] == {
        "n": 3,
        "win_rate": 0.333,
        "tie_rate": 0.333,
        "loss_rate": 0.333,
    }
    assert judged["judges"]["j2"]["esconv"]["win_rate"] == 1.0
    assert judged["agreement"]["all_agree_rate"] == round(2 / 3, 3)
    assert judged["agreement"]["combined"]["all"]["n"] == 2

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "a", "preference": "C"}]))
    args.judgements = [str(bad)]
    with pytest.raises(SystemExit):
        runner.score(args)

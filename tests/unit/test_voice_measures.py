from __future__ import annotations

from app.evaluation.voice import summarize, surface


def test_surface_flags_the_robotic_patterns() -> None:
    reply = (
        f"It sounds like that{chr(0x2019)}s rough. Here are a few ideas:\n\n"
        "* **Write it down** - a journal\n* Talk to someone\n\n"
        "Would any of those work (e.g. tonight)?"
    )
    m = surface(reply, "ugh, rough day")
    assert m["list_lines"] == 2 and m["bold_or_heading"] == 1
    assert m["ends_with_question"] and m["option_menu"]
    assert m["stock_phrases"] == ["it sounds like", "here are a few", "a few ideas"]
    assert m["message_words"] == 3


def test_a_plain_reply_passes_and_summaries_are_rates() -> None:
    plain = surface("Ugh. Twelve-hour shift and a broken printer? You deserve a sofa.", "ugh")
    assert plain["list_lines"] == 0 and not plain["ends_with_question"]
    assert not plain["stock_phrases"]
    summary = summarize([plain, surface("How are you?", "hi")])
    assert summary["replies"] == 2 and summary["ending_with_question"] == 0.5
    assert summarize([]) == {"replies": 0}

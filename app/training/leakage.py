"""Held-out isolation (spec §16.3, §17.3 split/leakage checks).

Two guards keep evaluation data out of training data:

- **conversation ids**: a source conversation that any evaluation item
  was drawn from is never read for training (the dialogue benchmark items,
  and the opening messages of the companion 1.5.0 voice held-out set);
- **text overlap**: every text of at least `MIN_WORDS` words in every
  evaluation file is indexed, and a training example whose message,
  history or target repeats one exactly, or shares an `NGRAM`-word run with
  one, is dropped as a leak.

The text index is built from every string in every evaluation file, so it
does not depend on each file's field names.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.rag.extract import normalize_text

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval" / "datasets"
VOICE_HELD_OUT = ROOT / "eval" / "results" / "voice-companion-1.4.0-vs-1.5.0-20261015.json"
MIN_WORDS = 4
NGRAM = 8
_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text).lower().replace("\u2019", "'"))


def _key(text: str) -> str:
    return hashlib.sha256(" ".join(_words(text)).encode()).hexdigest()


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from _strings(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _strings(inner)


@dataclass
class LeakageIndex:
    texts: set[str] = field(default_factory=set)
    ngrams: set[tuple[str, ...]] = field(default_factory=set)
    # (source id, conversation id) pairs used by evaluation.
    conversations: set[tuple[str, str]] = field(default_factory=set)
    files: dict[str, str] = field(default_factory=dict)  # path -> sha256

    def add_text(self, text: str) -> None:
        words = _words(text)
        if len(words) < MIN_WORDS:
            return
        self.texts.add(_key(text))
        for i in range(len(words) - NGRAM + 1):
            self.ngrams.add(tuple(words[i : i + NGRAM]))

    def leaks(self, text: str) -> bool:
        words = _words(text)
        if len(words) < MIN_WORDS:
            return False
        if _key(text) in self.texts:
            return True
        return any(
            tuple(words[i : i + NGRAM]) in self.ngrams for i in range(len(words) - NGRAM + 1)
        )


def _read_jsonl(path: Path) -> Iterable[Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def build_index(eval_dir: Path = EVAL_DIR, voice_results: Path = VOICE_HELD_OUT) -> LeakageIndex:
    """Index every evaluation file under `eval_dir` (committed sets and the
    downloaded `external/` ones), and the conversation ids evaluation used."""
    index = LeakageIndex()
    for path in sorted(eval_dir.rglob("*.jsonl")):
        index.files[str(path.relative_to(eval_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in _read_jsonl(path):
            for text in _strings(row):
                index.add_text(text)
            meta = row.get("meta") if isinstance(row, dict) else None
            if isinstance(meta, dict) and "conversation" in meta and "source" in row:
                index.conversations.add((str(row["source"]), str(meta["conversation"])))
    if voice_results.exists():
        judged = json.loads(voice_results.read_text(encoding="utf-8"))
        for item_id in judged.get("sets", {}).get("held_out_public", {}).get("judged", {}):
            prefix, _, conversation = str(item_id).partition("-")
            source = {"esconv": "esconv", "ed": "empatheticdialogues"}.get(prefix)
            if source:
                index.conversations.add((source, conversation))
    return index

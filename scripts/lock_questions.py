"""Record checksums for new decision-question versions in decisions/questions.lock.json.

Existing entries are never changed: if a released question was edited, this
script fails and the change must be published as a new version instead.

Usage: uv run python scripts/lock_questions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.decisions.questions import LOCK_FILE, QUESTIONS, load_lock


def main() -> int:
    lock = load_lock()
    added: list[str] = []
    for question in QUESTIONS:
        existing = lock.get(question.key)
        if existing is None:
            lock[question.key] = question.checksum
            added.append(question.key)
        elif existing != question.checksum:
            print(f"ERROR: {question.key} was modified after release; bump its version.")
            return 1
    if added:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": keep LF on Windows so git does not see a rewritten file.
        LOCK_FILE.write_text(
            json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print("Locked:", ", ".join(added) if added else "nothing new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

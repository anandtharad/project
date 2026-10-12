"""Record checksums for new prompt versions in prompts/prompts.lock.json.

Existing entries are never changed: if a released template was edited, this
script fails and the change must be published as a new version instead.

Usage: uv run python scripts/lock_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompting import LOCK_FILE_NAME, PROMPTS_DIR, discover_prompts, load_lock


def main() -> int:
    lock = load_lock()
    added: list[str] = []
    for prompt in discover_prompts():
        existing = lock.get(prompt.key)
        if existing is None:
            lock[prompt.key] = prompt.checksum
            added.append(prompt.key)
        elif existing != prompt.checksum:
            print(f"ERROR: {prompt.key} was modified after release; bump its version.")
            return 1
    if added:
        # newline="\n": keep LF on Windows so git does not see a rewritten file.
        (PROMPTS_DIR / LOCK_FILE_NAME).write_text(
            json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print("Locked:", ", ".join(added) if added else "nothing new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

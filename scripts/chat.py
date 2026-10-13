"""Minimal terminal client for manual testing of a running API (not a frontend).

Usage:
    uv run python scripts/chat.py --token <dev token> [--url http://localhost:8000]

Quit with Ctrl-D (Ctrl-Z then Enter on Windows) or Ctrl-C.

Each message is sent with a fresh idempotency key, so a network retry of the
same message never creates a duplicate turn.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--token", required=True, help="a token from DEV_AUTH_TOKENS")
    parser.add_argument("--conversation", help="continue an existing conversation ID")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.token}"}
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        conversation_id = args.conversation
        if conversation_id is None:
            created = client.post("/v1/conversations")
            if created.status_code == 401:
                print(
                    "401 Unauthorized: the server does not know this token.\n"
                    "--token is the text after the colon in DEV_AUTH_TOKENS in the server's\n"
                    ".env (for 'me:abc123' pass --token abc123). After editing .env, restart\n"
                    "the server. 'uv run python scripts/dev.py doctor' lists the configured users.",
                    file=sys.stderr,
                )
                return 1
            created.raise_for_status()
            conversation_id = created.json()["conversation_id"]
        quit_keys = "Ctrl-Z then Enter" if os.name == "nt" else "Ctrl-D"
        print(f"conversation {conversation_id} - {quit_keys} (or Ctrl-C) to quit\n")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            # A stray ^D (Windows) or ^Z line is a quit attempt, not a message.
            if text in ("\x04", "\x1a"):
                return 0
            if not text:
                continue
            key = uuid.uuid4().hex
            for attempt in range(2):
                try:
                    response = client.post(
                        f"/v1/conversations/{conversation_id}/messages",
                        json={"content": text, "idempotency_key": key},
                    )
                    break
                except httpx.TransportError:
                    if attempt == 1:
                        raise
            if response.status_code in (200, 201):
                print(f"companion> {response.json()['assistant_content']}\n")
            else:
                error = response.json().get("error", {})
                print(f"[{response.status_code} {error.get('code')}] {error.get('message')}\n")


if __name__ == "__main__":
    sys.exit(main())

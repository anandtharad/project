"""Minimal terminal client for manual testing of a running API (not a frontend).

Usage:
    uv run python scripts/chat.py --token <dev token> [--url http://localhost:8000]

Each message is sent with a fresh idempotency key, so a network retry of the
same message never creates a duplicate turn.
"""

from __future__ import annotations

import argparse
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
            created.raise_for_status()
            conversation_id = created.json()["conversation_id"]
        print(f"conversation {conversation_id} - Ctrl-D to quit\n")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                print()
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

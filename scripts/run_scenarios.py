"""Run the scripted conversation scenarios against a running server and
write a report to check by hand (eval/datasets/conversation_scenarios).

    uv run python scripts/run_scenarios.py --url http://localhost:8001 --token testtoken
    uv run python scripts/run_scenarios.py --url http://localhost:8001 --token testtoken \
        --only stm-01-interview-detour grief-03-nani-anniversary --label agent-on

WARNING: for each scenario the runner deletes ALL memories of the user
behind the token and changes their memory settings. Use a test user, e.g.
add `tester:testtoken` to DEV_AUTH_TOKENS in the server's .env and restart.

For each scenario it:
1. sets the scenario's memory settings and forgets all memories;
2. runs each session as a new conversation, message by message;
3. after each session, waits for the background memory step, then records
   the stored memories (superseded ones too) and, when the database in
   .env is reachable, the conversation's notes (short-term state);
4. runs the actions between sessions (deleting matching memories);
5. checks the expectations (word matches, hints only);
6. deletes the conversations it created (--keep to keep them) and resets
   the memory settings.

Writes eval/results/scenarios-<label>-<time>.md (to read) and .json.
Free-tier note: every message costs a reply call, plus state, memory and
safety calls; use --only, and consider STATE_EXTRACTION_MODEL and
MEMORY_EXTRACTION_MODEL=openai/gpt-oss-20b on the server to spread the
daily token budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation.scenarios import (  # noqa: E402
    DATASET,
    Flag,
    Scenario,
    check_memories,
    check_recall,
    load,
)

RESULTS_DIR = ROOT / "eval" / "results"


class StateReader:
    """Reads a conversation's notes from the database in .env (optional)."""

    def __init__(self) -> None:
        self._engine: Any = None
        self.error: str | None = None

    async def start(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            from app.config import Settings

            self._engine = create_async_engine(Settings().database_url)
            async with self._engine.connect() as conn:
                await conn.exec_driver_sql("SELECT 1")
        except Exception as exc:  # the report just says notes are unavailable
            self.error = type(exc).__name__
            self._engine = None

    async def notes(self, conversation_id: str) -> dict[str, Any] | None:
        if self._engine is None:
            return None
        from sqlalchemy import text

        async with self._engine.connect() as conn:
            value = await conn.scalar(
                text("SELECT state FROM conversation_states WHERE conversation_id = :c"),
                {"c": uuid.UUID(conversation_id)},
            )
        if not isinstance(value, dict):
            return None
        return {k: v for k, v in value.items() if v not in (None, "", [], {})}

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()


async def send(
    client: httpx.AsyncClient, conversation: str, content: str, pause: float
) -> dict[str, Any]:
    key = uuid.uuid4().hex
    for attempt in range(5):
        response = await client.post(
            f"/v1/conversations/{conversation}/messages",
            json={"content": content, "idempotency_key": key},
        )
        if response.status_code in (200, 201):
            body = response.json()
            return {
                "status": response.status_code,
                "reply": body["assistant_content"],
                "pathway": (body.get("safety") or {}).get("pathway", "normal"),
                "sources": [s.get("title") for s in body.get("sources") or []],
            }
        retryable = response.status_code in (409, 429, 503)
        if not retryable or attempt == 4:
            error = response.json().get("error", {}) if response.content else {}
            return {"status": response.status_code, "reply": None, "error": error.get("code")}
        wait = float(response.headers.get("Retry-After") or 0) or 15.0 * (attempt + 1)
        print(f"    {response.status_code}, retrying in {wait:.0f}s", flush=True)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def memories(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    response = await client.get("/v1/users/me/memories", params={"status": "all"})
    response.raise_for_status()
    found: list[dict[str, Any]] = response.json()["memories"]
    return found


async def run_scenario(
    client: httpx.AsyncClient,
    scenario: Scenario,
    states: StateReader,
    *,
    pause: float,
    settle: float,
    keep: bool,
) -> dict[str, Any]:
    settings = scenario.settings.model_dump()
    (await client.put("/v1/users/me/memory-settings", json=settings)).raise_for_status()
    (await client.delete("/v1/users/me/memories")).raise_for_status()
    flags: list[Flag] = []
    sessions: list[dict[str, Any]] = []
    created: list[str] = []
    for number, session in enumerate(scenario.sessions, 1):
        made = await client.post("/v1/conversations")
        made.raise_for_status()
        conversation = str(made.json()["conversation_id"])
        created.append(conversation)
        turns: list[dict[str, Any]] = []
        for index, message in enumerate(session.messages):
            print(f"  chat {number}, message {index + 1}/{len(session.messages)}", flush=True)
            turn = await send(client, conversation, message, pause)
            turns.append({"you": message, **turn})
            await asyncio.sleep(pause)
        for check in session.recall:
            where = f"chat {number}, message {check.message + 1}"
            flags += check_recall(check, turns[check.message].get("reply"), where)
        await asyncio.sleep(settle)  # the memory step runs after each reply
        stored = await memories(client)
        actions: list[str] = []
        if session.after is not None:
            for word in session.after.delete_memories_matching:
                for memory in stored:
                    if memory["status"] != "superseded" and word.casefold() in (
                        memory["content"].casefold()
                    ):
                        await client.delete(f"/v1/users/me/memories/{memory['memory_id']}")
                        actions.append(f"deleted memory {memory['content']!r}")
        sessions.append(
            {
                "conversation_id": conversation,
                "turns": turns,
                "memories_after": stored,
                "notes": await states.notes(conversation),
                "actions": actions,
            }
        )
    final = await memories(client)
    flags += check_memories(scenario.expect, final)
    if not keep:
        for conversation in created:
            await client.delete(f"/v1/conversations/{conversation}")
    await client.put(
        "/v1/users/me/memory-settings", json={"memory_enabled": True, "allow_sensitive": False}
    )
    return {
        "id": scenario.id,
        "theme": scenario.theme,
        "check": scenario.check,
        "settings": settings,
        "sessions": sessions,
        "final_memories": final,
        "flags": [{"kind": f.kind, "ok": f.ok, "detail": f.detail} for f in flags],
    }


def _memory_line(memory: dict[str, Any]) -> str:
    return f"- [{memory['memory_type']}, {memory['status']}] {memory['content']}"


def markdown(results: list[dict[str, Any]], label: str, state_error: str | None) -> str:
    out = [f"# Scenario run: {label}", ""]
    if state_error:
        out += [f"Conversation notes not shown (database not reachable: {state_error}).", ""]
    problems = sum(1 for r in results for f in r["flags"] if not f["ok"])
    out += [
        f"{len(results)} scenarios, {problems} automatic flags to look at. The flags are word "
        "matches: read the transcripts before deciding.",
        "",
    ]
    for result in results:
        bad = [f for f in result["flags"] if not f["ok"]]
        out += [
            f"## {result['id']} {'⚠️' if bad else '✅'}",
            "",
            f"**Theme:** {result['theme']}",
            "",
            f"**What to check:** {result['check']}",
            "",
            f"**Memory settings:** {result['settings']}",
            "",
        ]
        for number, session in enumerate(result["sessions"], 1):
            out += [f"### Chat {number}", ""]
            for turn in session["turns"]:
                out.append(f"> **you:** {turn['you']}")
                out.append(">")
                if turn.get("reply") is None:
                    out.append(f"> **error:** {turn['status']} {turn.get('error')}")
                else:
                    pathway = "" if turn["pathway"] == "normal" else f" *[{turn['pathway']}]*"
                    reply = turn["reply"].replace("\n", "\n> ")
                    out.append(f"> **companion:**{pathway} {reply}")
                    if turn.get("sources"):
                        out.append(f"> *sources: {', '.join(turn['sources'])}*")
                out.append("")
            notes = session.get("notes")
            if notes:
                out += [
                    "**Short-term notes for this chat:** `"
                    + json.dumps(notes, ensure_ascii=False)
                    + "`",
                    "",
                ]
            stored = session["memories_after"]
            out.append("**Long-term memories after this chat:**")
            out += [_memory_line(m) for m in stored] or ["- (none)"]
            out += [f"- *{a}*" for a in session["actions"]]
            out.append("")
        out.append("**Automatic checks:**")
        out += [f"- {'✅' if f['ok'] else '⚠️'} {f['detail']}" for f in result["flags"]] or [
            "- (none)"
        ]
        out.append("")
    return "\n".join(out) + "\n"


async def run(args: argparse.Namespace) -> int:
    dataset = load(Path(args.dataset))
    scenarios = [s for s in dataset.scenarios if not args.only or s.id in set(args.only)]
    if not scenarios:
        print("no scenarios match --only", file=sys.stderr)
        return 1
    states = StateReader()
    if not args.no_db:
        await states.start()
    results: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {args.token}"}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=180) as client:
        ready = await client.get("/ready")
        if ready.status_code != 200:
            print(f"the server is not ready ({ready.status_code}): is the database up?")
            return 1
        check = await client.get("/v1/users/me/memories")
        if check.status_code == 401:
            print("401: the server does not know this token (DEV_AUTH_TOKENS).", file=sys.stderr)
            return 1
        for n, scenario in enumerate(scenarios, 1):
            print(f"[{n}/{len(scenarios)}] {scenario.id}", flush=True)
            results.append(
                await run_scenario(
                    client,
                    scenario,
                    states,
                    pause=args.pause,
                    settle=args.settle,
                    keep=args.keep,
                )
            )
    await states.close()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"scenarios-{args.label}-{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(
            {"label": args.label, "dataset_version": dataset.version, "results": results},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    base.with_suffix(".md").write_text(
        markdown(results, args.label, states.error), encoding="utf-8"
    )
    problems = sum(1 for r in results for f in r["flags"] if not f["ok"])
    print(f"done: {problems} automatic flags. Read {base.with_suffix('.md').relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--token", required=True, help="a TEST user's token from DEV_AUTH_TOKENS")
    parser.add_argument("--only", nargs="*", help="scenario ids")
    parser.add_argument("--label", default="run", help="a name for the report, e.g. agent-on")
    parser.add_argument("--pause", type=float, default=5.0, help="seconds after each message")
    parser.add_argument(
        "--settle", type=float, default=12.0, help="seconds after each chat for the memory step"
    )
    parser.add_argument("--keep", action="store_true", help="keep the conversations")
    parser.add_argument("--no-db", action="store_true", help="don't read conversation notes")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--dataset", default=str(DATASET))
    args = parser.parse_args()
    if not args.yes:
        answer = input(
            "This deletes ALL memories of the user behind this token, before every scenario, "
            "and changes their memory settings. Use a test user. Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            return 1
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""The voice WebSocket end to end (app/api/voice.py) with the fake speech
providers: tickets, a spoken turn through the real orchestrator and
database, typed input with text-only output, and refused connections."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
import websockets
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.helpers import (
    ALICE,
    BOB,
    ScriptedLLMClient,
    create_conversation,
    make_response,
    make_settings,
    running_app,
)

PCM = b"\x10\x00" * 1600  # 0.1 s of 16 kHz audio


@contextlib.asynccontextmanager
async def voice_server(
    database_url: str,
    llm: ScriptedLLMClient,
    unvalidated: dict[str, Any] | None = None,
    **overrides: Any,
) -> AsyncIterator[Any]:
    settings = make_settings(
        database_url=database_url,
        voice_enabled=True,
        stt_provider="fake",
        tts_provider="fake",
        **overrides,
    )
    if unvalidated:  # values below the settings' minimums, for fast tests
        settings = settings.model_copy(update=unvalidated)
    async with running_app(settings, llm) as (app, client):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning")
        )
        task = asyncio.create_task(server.serve())
        while not server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
            await asyncio.sleep(0.01)
        try:
            yield app, client, f"ws://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await task


async def ticket(client: Any, conversation_id: str, headers: dict[str, str]) -> str:
    response = await client.post(
        f"/v1/conversations/{conversation_id}/voice/tickets", headers=headers
    )
    assert response.status_code == 201, response.text
    return str(response.json()["ticket"])


async def receive_until(ws: Any, kind: str, audio: list[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        frame = await asyncio.wait_for(ws.recv(), 10)
        if isinstance(frame, bytes):
            audio.append(frame)
            continue
        event = json.loads(frame)
        events.append(event)
        if event["type"] == kind:
            return events


async def test_a_spoken_turn_end_to_end(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    llm = ScriptedLLMClient()
    llm.queue(make_response("That sounds like a lot. What happened?"))
    async with voice_server(database_url, llm) as (_, client, base):
        cid = await create_conversation(client, ALICE)
        async with websockets.connect(
            f"{base}/v1/voice?ticket={await ticket(client, cid, ALICE)}"
        ) as ws:
            await ws.send(json.dumps({"type": "start", "output": ["text", "audio"]}))
            audio: list[bytes] = []
            ready = await receive_until(ws, "ready", audio)
            assert ready[-1]["input"] == ["text", "audio"] and ready[-1]["output"] == [
                "text",
                "audio",
            ]
            for _ in range(3):
                await ws.send(PCM)
            await ws.send(json.dumps({"type": "end_of_speech"}))
            events = await receive_until(ws, "audio_end", audio)
            transcript = [e for e in events if e["type"] == "transcript" and e["final"]]
            reply = next(e for e in events if e["type"] == "reply")
            assert transcript[0]["text"] == "(0.3 seconds of audio)"
            assert reply["text"] == "That sounds like a lot. What happened?"
            assert audio, "the reply was spoken"
            await ws.send(json.dumps({"type": "playback_done"}))
            await receive_until(ws, "state", audio)  # listening again
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT role, content, generation_metadata FROM messages"
                    " WHERE conversation_id = :c ORDER BY sequence"
                ),
                {"c": uuid.UUID(cid)},
            )
        ).all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[0].content == "(0.3 seconds of audio)"
    assert rows[1].generation_metadata["input"] == {
        "modality": "voice",
        "channel": "voice_socket",
        "stt": {"provider": "fake", "model": "fake-stt"},
    }


async def test_typed_input_with_text_only_output(database_url: str, clean_db: None) -> None:
    llm = ScriptedLLMClient()
    llm.queue(make_response("Hi! How's your day going?"))
    async with voice_server(database_url, llm) as (_, client, base):
        cid = await create_conversation(client, ALICE)
        async with websockets.connect(
            f"{base}/v1/voice?ticket={await ticket(client, cid, ALICE)}"
        ) as ws:
            await ws.send(json.dumps({"type": "start", "output": ["text"]}))
            audio: list[bytes] = []
            ready = await receive_until(ws, "ready", audio)
            assert ready[-1]["output"] == ["text"]
            await ws.send(json.dumps({"type": "text", "content": "hey"}))
            events = await receive_until(ws, "reply", audio)
            assert events[-1]["text"] == "Hi! How's your day going?"
            await receive_until(ws, "state", audio)
            assert audio == []


async def test_tickets_are_owned_single_use_and_required(database_url: str, clean_db: None) -> None:
    llm = ScriptedLLMClient()
    async with voice_server(database_url, llm) as (_, client, base):
        cid = await create_conversation(client, ALICE)
        # Another user's conversation: same 404 as a missing one.
        foreign = await client.post(f"/v1/conversations/{cid}/voice/tickets", headers=BOB)
        assert foreign.status_code == 404
        good = await ticket(client, cid, ALICE)
        async with websockets.connect(f"{base}/v1/voice?ticket={good}") as ws:
            await ws.send(json.dumps({"type": "stop"}))
        for bad in (good, "not-a-ticket", ""):
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                async with websockets.connect(f"{base}/v1/voice?ticket={bad}") as ws:
                    await ws.recv()
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 4401


async def test_voice_disabled(database_url: str, clean_db: None) -> None:
    llm = ScriptedLLMClient()
    async with running_app(make_settings(database_url=database_url), llm) as (_, client):
        cid = await create_conversation(client, ALICE)
        response = await client.post(f"/v1/conversations/{cid}/voice/tickets", headers=ALICE)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "voice_disabled"


async def closed_code(ws: Any) -> int | None:
    """Read until the server closes; its close code."""
    with contextlib.suppress(websockets.exceptions.ConnectionClosed):
        while True:
            await asyncio.wait_for(ws.recv(), 10)
    return ws.close_code  # type: ignore[no-any-return]


async def test_sessions_per_person_are_limited(database_url: str, clean_db: None) -> None:
    llm = ScriptedLLMClient()
    async with voice_server(database_url, llm, voice_max_sessions_per_user=1) as (_, client, base):
        cid = await create_conversation(client, ALICE)
        spare = await ticket(client, cid, ALICE)  # issued while no session is open
        async with websockets.connect(
            f"{base}/v1/voice?ticket={await ticket(client, cid, ALICE)}"
        ) as ws:
            await receive_until(ws, "ready", [])
            refused = await client.post(f"/v1/conversations/{cid}/voice/tickets", headers=ALICE)
            assert refused.status_code == 429
            assert refused.json()["error"]["code"] == "too_many_voice_sessions"
            async with websockets.connect(f"{base}/v1/voice?ticket={spare}") as second:
                assert await closed_code(second) == 4429
            # Another person is not affected.
            bob_cid = await create_conversation(client, BOB)
            await ticket(client, bob_cid, BOB)
        for _ in range(200):  # once the server has seen the close
            again = await client.post(f"/v1/conversations/{cid}/voice/tickets", headers=ALICE)
            if again.status_code == 201:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the session was never released")


async def test_frames_validation_stop_and_idle(
    database_url: str, clean_db: None, db_engine: AsyncEngine
) -> None:
    llm = ScriptedLLMClient()
    llm.queue(make_response("Still here."))
    async with voice_server(database_url, llm, unvalidated={"voice_idle_timeout_seconds": 1}) as (
        _,
        client,
        base,
    ):
        cid = await create_conversation(client, ALICE)
        # An oversized first frame is not parsed: the default outputs apply,
        # and the session works.
        huge = json.dumps({"type": "start", "output": ["text"] + [0] * 40_000})
        async with websockets.connect(
            f"{base}/v1/voice?ticket={await ticket(client, cid, ALICE)}", max_size=None
        ) as ws:
            await ws.send(huge)
            ready = await receive_until(ws, "ready", [])
            assert ready[-1]["output"] == ["text", "audio"]
            # Text over the socket follows the HTTP API's rules.
            await ws.send(json.dumps({"type": "text", "content": "hello\u0000world"}))
            error = await receive_until(ws, "error", [])
            assert error[-1]["code"] == "validation_failed"
            await ws.send(json.dumps({"type": "start", "output": ["text"]}))
            await receive_until(ws, "error", [])  # "start" is only read first
            # No message from the client for the idle timeout: closed.
            assert await closed_code(ws) == 4408
        async with db_engine.connect() as conn:
            stored = (await conn.execute(text("SELECT count(*) FROM messages"))).scalar_one()
        assert stored == 0
        # A first "stop" ends the session.
        async with websockets.connect(
            f"{base}/v1/voice?ticket={await ticket(client, cid, ALICE)}"
        ) as ws:
            await ws.send(json.dumps({"type": "stop"}))
            assert await closed_code(ws) == 1000

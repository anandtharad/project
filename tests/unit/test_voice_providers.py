"""Speech provider adapters against local servers that follow each
provider's documented protocol (docs/voice-interface.md). These check what
the adapters send and how they read replies; they are not live checks of
the providers."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from app.voice.base import AudioFormat, SpeechError, TranscriptKind
from app.voice.providers import deepgram, gnani, sarvam

AUDIO = AudioFormat(sample_rate=16_000)
PCM = b"\x01\x00" * 800  # 1600 bytes


@contextlib.asynccontextmanager
async def server(
    handler: Callable[[ServerConnection], Awaitable[None]],
) -> AsyncIterator[tuple[str, list[Any]]]:
    seen: list[Any] = []

    async def wrapped(ws: ServerConnection) -> None:
        seen.append(("path", ws.request.path if ws.request else None))
        # HTTP header names are case-insensitive.
        headers = {k.lower(): v for k, v in ws.request.headers.raw_items()} if ws.request else {}
        seen.append(("headers", headers))
        await handler(ws)

    async with serve(wrapped, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        wrapped.seen = seen  # type: ignore[attr-defined]
        yield f"ws://127.0.0.1:{port}", seen


def collector(seen: list[Any]) -> Callable[[ServerConnection], Awaitable[Any]]:
    async def receive(ws: ServerConnection) -> Any:
        message = await ws.recv()
        seen.append(message if isinstance(message, bytes) else json.loads(message))
        return seen[-1]

    return receive


async def events_of(stream: Any) -> list[tuple[TranscriptKind, str]]:
    return [(e.kind, e.text) async for e in stream.events()]


# --- Deepgram -------------------------------------------------------------


async def test_deepgram_stt_streams_binary_and_finalizes() -> None:
    seen: list[Any] = []

    async def handler(ws: ServerConnection) -> None:
        receive = collector(seen)
        assert await receive(ws) == PCM
        await ws.send(json.dumps({"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.1}))
        await ws.send(
            json.dumps(
                {
                    "type": "Results",
                    "is_final": False,
                    "channel": {"alternatives": [{"transcript": "hello"}]},
                }
            )
        )
        assert await receive(ws) == {"type": "Finalize"}
        assert await receive(ws) == {"type": "CloseStream"}
        await ws.send(
            json.dumps(
                {
                    "type": "Results",
                    "is_final": True,
                    "speech_final": True,
                    "channel": {
                        "alternatives": [{"transcript": "hello there", "confidence": 0.97}]
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "Metadata", "request_id": "r"}))
        await ws.close()

    async with server(handler) as (url, info):
        stt = deepgram.DeepgramSpeechToText("dg-key", base=url)
        stream = await stt.open(AUDIO, "hi-IN")
        await stream.send(PCM)
        await stream.send(b"")  # never sent: an empty frame closes the stream
        reader = asyncio.create_task(events_of(stream))
        await asyncio.sleep(0.05)
        await stream.finish()
        events = await reader
    assert events == [
        (TranscriptKind.SPEECH_STARTED, ""),
        (TranscriptKind.INTERIM, "hello"),
        (TranscriptKind.FINAL, "hello there"),
    ]
    path = dict(info)["path"]
    assert path.startswith("/v1/listen?") and "language=hi&" in path and "encoding=linear16" in path
    assert "sample_rate=16000" in path and "interim_results=true" in path
    assert dict(info)["headers"]["authorization"] == "Token dg-key"


async def test_deepgram_tts_speaks_flushes_and_clears() -> None:
    seen: list[Any] = []

    async def handler(ws: ServerConnection) -> None:
        receive = collector(seen)
        assert (await receive(ws))["type"] == "Speak"
        assert await receive(ws) == {"type": "Flush"}
        await ws.send(b"\x00\x01" * 100)
        await ws.send(json.dumps({"type": "Flushed", "sequence_id": 0}))
        with contextlib.suppress(Exception):
            assert await receive(ws) == {"type": "Close"}

    async with server(handler) as (url, info):
        tts = deepgram.DeepgramTextToSpeech("dg-key", base=url)
        stream = await tts.open(None, "en-IN")
        await stream.send_text("Hello there. ")
        await stream.flush()
        chunks = [c async for c in stream.audio()]
        await stream.close()
    assert chunks and chunks[0].format.sample_rate == 24_000
    assert seen[0] == {"type": "Speak", "text": "Hello there. "}
    path = dict(info)["path"]
    assert "model=aura-2-thalia-en" in path and "encoding=linear16" in path

    # Barge-in: Clear is sent and no more audio is yielded.
    cleared: list[Any] = []

    async def clear_handler(ws: ServerConnection) -> None:
        receive = collector(cleared)
        await receive(ws)  # Speak
        await receive(ws)  # Flush
        await ws.send(b"\x00\x01" * 100)
        assert await receive(ws) == {"type": "Clear"}
        await ws.send(b"\x00\x01" * 100)  # in flight when the clear arrived
        await ws.send(json.dumps({"type": "Cleared", "sequence_id": 0}))

    async with server(clear_handler) as (url, _):
        stream = await deepgram.DeepgramTextToSpeech("k", base=url).open(None, "en-IN")
        await stream.send_text("A long reply. ")
        await stream.flush()
        got = []
        async for chunk in stream.audio():
            got.append(chunk)
            await stream.cancel()
    assert len(got) == 1


async def test_failed_handshakes_are_speech_errors() -> None:
    async def reject(ws: ServerConnection) -> None:
        await ws.close(code=4001, reason="auth")

    async with server(reject) as (url, _):
        stream = await deepgram.DeepgramSpeechToText("k", base=url).open(AUDIO, "en-IN")
        with pytest.raises(SpeechError):
            await events_of(stream)
    with pytest.raises(SpeechError):
        await deepgram.DeepgramSpeechToText("k", base="ws://127.0.0.1:9").open(AUDIO, "en-IN")


# --- Sarvam ---------------------------------------------------------------


async def test_sarvam_stt_manual_endpointing_and_base64_audio() -> None:
    seen: list[Any] = []

    async def handler(ws: ServerConnection) -> None:
        receive = collector(seen)
        assert await receive(ws) == {"event": "speech_start"}
        first = await receive(ws)
        assert first["event"] == "audio_input" and base64.b64decode(first["audio"]) == PCM
        await ws.send(
            json.dumps({"event": "transcript.partial", "utterance_idx": 0, "text": "namaste"})
        )
        assert await receive(ws) == {"event": "speech_end"}
        assert await receive(ws) == {"event": "end"}
        await ws.send(
            json.dumps({"event": "transcript.final", "utterance_idx": 0, "text": "namaste ji"})
        )
        await ws.send(json.dumps({"event": "session.end", "request_id": "r"}))

    async with server(handler) as (url, info):
        stt = sarvam.SarvamSpeechToText("sv-key", base=url)
        stream = await stt.open(AUDIO, "hi-IN")
        await stream.send(PCM)
        reader = asyncio.create_task(events_of(stream))
        await asyncio.sleep(0.05)
        await stream.finish()
        events = await reader
    assert events == [(TranscriptKind.INTERIM, "namaste"), (TranscriptKind.FINAL, "namaste ji")]
    path = dict(info)["path"]
    assert path.startswith("/speech-to-text-realtime/ws?") and "language_code=hi-IN" in path
    assert "endpointing=manual" in path
    assert dict(info)["headers"]["api-subscription-key"] == "sv-key"
    with pytest.raises(SpeechError):
        await stt.open(AudioFormat(sample_rate=48_000), "hi-IN")


async def test_sarvam_tts_config_text_flush_and_formats() -> None:
    seen: list[Any] = []

    async def handler(ws: ServerConnection) -> None:
        receive = collector(seen)
        config = await receive(ws)
        assert config["type"] == "config" and config["data"]["output_audio_codec"] == "linear16"
        assert (await receive(ws))["type"] == "text"
        assert await receive(ws) == {"type": "flush"}
        wav = (
            b"RIFF\x00\x00\x00\x00WAVEfmt " + bytes(16) + b"data\x08\x00\x00\x00" + b"\x01\x02" * 4
        )
        raw = b"\x03\x04" * 4
        for chunk, kind in ((wav, "audio/wav"), (raw, "audio/l16")):
            payload = {
                "content_type": kind,
                "audio": base64.b64encode(chunk).decode(),
                "request_id": "r",
            }
            await ws.send(json.dumps({"type": "audio", "data": payload}))
        await ws.send(json.dumps({"type": "event", "data": {"event_type": "final"}}))

    async with server(handler) as (url, info):
        stream = await sarvam.SarvamTextToSpeech("sv-key", base=url).open("anushka", "hi-IN")
        await stream.send_text("Namaste. ")
        await stream.flush()
        chunks = [c.pcm async for c in stream.audio()]
    assert chunks == [b"\x01\x02" * 4, b"\x03\x04" * 4]  # the WAV header was removed
    assert seen[0]["data"]["speaker"] == "anushka" and seen[0]["data"]["language_code"] == "hi-IN"
    assert "model=bulbul%3Av4-flash" in dict(info)["path"]
    with pytest.raises(SpeechError, match="compressed"):
        sarvam.pcm_from(b"\xff\xfb\x90\x00", "audio/mp3")


# --- Gnani ----------------------------------------------------------------


async def test_gnani_stt_exact_frames_trailing_silence_and_finals() -> None:
    frames: list[bytes] = []

    async def handler(ws: ServerConnection) -> None:
        await ws.send(
            json.dumps({"type": "connected", "config": {"sample_rate": 16000, "chunk_size": 512}})
        )
        silent = 0
        async for message in ws:
            assert isinstance(message, bytes)
            frames.append(message)
            silent = silent + 1 if message == bytes(1024) else 0
            if silent == 5:
                await ws.send(json.dumps({"type": "processing"}))
                await ws.send(
                    json.dumps(
                        {"type": "transcript", "text": "hello from gnani", "segment_index": 0}
                    )
                )

    async with server(handler) as (url, info):
        stt = gnani.GnaniSpeechToText("gn-key", url=url)
        stream = await stt.open(AUDIO, "en-IN")
        await stream.send(PCM)  # 1600 bytes: one frame now, 576 bytes kept
        reader = asyncio.create_task(events_of(stream))
        await stream.finish()
        events = await asyncio.wait_for(reader, 5)
        await stream.close()
    assert all(len(f) == 1024 for f in frames)
    assert frames[1][:576] == PCM[1024:]  # the last partial frame, padded
    assert events == [
        (TranscriptKind.UTTERANCE_END, ""),
        (TranscriptKind.FINAL, "hello from gnani"),
    ]
    headers = dict(info)["headers"]
    assert headers["x-api-key-id"] == "gn-key" and headers["lang_code"] == "en-IN"
    assert headers["x-sample-rate"] == "16000"


async def test_gnani_stt_ends_quietly_when_nothing_was_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gnani, "NO_SPEECH_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(gnani, "TRAILING_SILENCE_MS", 100)

    async def handler(ws: ServerConnection) -> None:
        async for _ in ws:
            pass

    async with server(handler) as (url, _):
        stream = await gnani.GnaniSpeechToText("k", url=url).open(AUDIO, "en-IN")
        reader = asyncio.create_task(events_of(stream))
        await stream.finish()
        assert await asyncio.wait_for(reader, 5) == []
        await stream.close()


async def test_gnani_tts_sends_one_request_per_sentence_and_reconnects() -> None:
    requests: list[dict[str, Any]] = []

    async def handler(ws: ServerConnection) -> None:
        # Like the official SDK's server: one request per connection.
        request = json.loads(await ws.recv())
        requests.append(request)
        await ws.send(json.dumps({"type": "start", "request_id": "r"}))
        audio = base64.b64encode(b"\x05\x06" * 10).decode()
        await ws.send(
            json.dumps(
                {"type": "audio", "data": {"chunk_index": 1, "audio": audio, "is_final": False}}
            )
        )
        await ws.send(json.dumps({"type": "complete", "data": {"audio": "", "is_final": True}}))
        await ws.close()

    async with server(handler) as (url, _):
        stream = await gnani.GnaniTextToSpeech("gn-key", url=url).open(None, "hi-IN")
        await stream.send_text("Pehla vakya. ")
        await stream.send_text("Doosra vakya. ")
        await stream.flush()
        chunks = [c async for c in stream.audio()]
    assert len(chunks) == 2
    assert [r["text"] for r in requests] == ["Pehla vakya.", "Doosra vakya."]
    assert requests[0]["voice"] == "Nalini" and requests[0]["model"] == "timbre-v2.5"
    assert requests[0]["audio_config"]["container"] == "raw"

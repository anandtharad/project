"""Deepgram streaming speech (checked against Deepgram's AsyncAPI reference,
guides and Python SDK source on 2026-10-16; docs/voice-interface.md).

Speech-to-text: wss://api.deepgram.com/v1/listen, configured by query
parameters, `Authorization: Token <key>`. Audio goes up as binary frames of
raw linear16; the server sends JSON `Results` (interim `is_final: false`,
then final), `SpeechStarted` (vad_events), `UtteranceEnd`
(utterance_end_ms), `Metadata` and `Error`. `Finalize` flushes buffered
audio; `CloseStream` makes the server send the last results and close.

Text-to-speech: wss://api.deepgram.com/v1/speak. `{"type": "Speak", "text"}`
adds text, `Flush` synthesises it (acknowledged by `Flushed`), audio comes
back as binary frames of raw linear16, `Clear` cancels for barge-in
(`Cleared`), `Close` ends the socket. At most 2000 characters per Speak.

Language: nova-3 takes en-IN and hi; Aura voices are English (and a few
European languages) only: no Hindi and no Indian-accent English.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from app.voice.base import AudioChunk, AudioFormat, SpeechError, TranscriptEvent, TranscriptKind
from app.voice.providers._ws import closed_error, open_socket, parse, send_json

DEFAULT_BASE = "wss://api.deepgram.com"
DEFAULT_STT_MODEL = "nova-3"
DEFAULT_TTS_MODEL = "aura-2-thalia-en"
TTS_SAMPLE_RATE = 24_000  # the streaming default
MAX_SPEAK_CHARACTERS = 2000
# BCP-47 -> Deepgram language codes where they differ.
LANGUAGES = {"hi-IN": "hi"}


class DeepgramSpeechToText:
    name = "deepgram"

    def __init__(self, api_key: str, *, model: str | None = None, base: str = DEFAULT_BASE) -> None:
        self._key = api_key
        self.model = model or DEFAULT_STT_MODEL
        self._base = base

    async def open(self, audio: AudioFormat, language: str) -> _DeepgramSttStream:
        query = urlencode(
            {
                "model": self.model,
                "language": LANGUAGES.get(language, language),
                "encoding": "linear16",
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "true",
                "vad_events": "true",
                "utterance_end_ms": 1000,
                "endpointing": 300,
            }
        )
        socket = await open_socket(
            self.name, f"{self._base}/v1/listen?{query}", {"Authorization": f"Token {self._key}"}
        )
        return _DeepgramSttStream(socket)


class _DeepgramSttStream:
    def __init__(self, socket: ClientConnection) -> None:
        self._socket = socket
        self._finished = False

    async def send(self, pcm: bytes) -> None:
        if not pcm:  # an empty frame closes the stream unexpectedly
            return
        try:
            await self._socket.send(pcm)
        except ConnectionClosed as exc:
            raise closed_error("deepgram", exc) from None

    async def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            await send_json(self._socket, {"type": "Finalize"})
            await send_json(self._socket, {"type": "CloseStream"})
        except ConnectionClosed as exc:
            raise closed_error("deepgram", exc) from None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        try:
            async for message in self._socket:
                event = parse(message)
                if event is None:
                    continue
                kind = event.get("type")
                if kind == "Results":
                    yield _result(event)
                elif kind == "SpeechStarted":
                    yield TranscriptEvent(TranscriptKind.SPEECH_STARTED)
                elif kind == "UtteranceEnd":
                    yield TranscriptEvent(TranscriptKind.UTTERANCE_END)
                elif kind == "Error":
                    raise SpeechError("deepgram", str(event.get("variant") or event.get("code")))
        except ConnectionClosed as exc:
            if exc.rcvd is None or exc.rcvd.code != 1000:
                raise closed_error("deepgram", exc) from None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._socket.close()


def _result(event: dict[str, Any]) -> TranscriptEvent:
    alternatives = (event.get("channel") or {}).get("alternatives") or [{}]
    best = alternatives[0]
    text = str(best.get("transcript") or "")
    confidence = best.get("confidence")
    kind = TranscriptKind.FINAL if event.get("is_final") else TranscriptKind.INTERIM
    return TranscriptEvent(kind, text, float(confidence) if confidence is not None else None)


class DeepgramTextToSpeech:
    name = "deepgram"
    output = AudioFormat(sample_rate=TTS_SAMPLE_RATE)

    def __init__(self, api_key: str, *, model: str | None = None, base: str = DEFAULT_BASE) -> None:
        self._key = api_key
        self.model = model or DEFAULT_TTS_MODEL
        self._base = base

    async def open(self, voice: str | None, language: str) -> _DeepgramTtsStream:
        # The voice is part of the model name (aura-2-<voice>-en).
        query = urlencode(
            {"model": voice or self.model, "encoding": "linear16", "sample_rate": TTS_SAMPLE_RATE}
        )
        socket = await open_socket(
            self.name, f"{self._base}/v1/speak?{query}", {"Authorization": f"Token {self._key}"}
        )
        return _DeepgramTtsStream(socket, self.output)


class _DeepgramTtsStream:
    def __init__(self, socket: ClientConnection, output: AudioFormat) -> None:
        self._socket = socket
        self._output = output
        self._cancelled = False

    async def send_text(self, text: str) -> None:
        try:
            for start in range(0, len(text), MAX_SPEAK_CHARACTERS):
                chunk = text[start : start + MAX_SPEAK_CHARACTERS]
                await send_json(self._socket, {"type": "Speak", "text": chunk})
        except ConnectionClosed as exc:
            raise closed_error("deepgram", exc) from None

    async def flush(self) -> None:
        try:
            await send_json(self._socket, {"type": "Flush"})
        except ConnectionClosed as exc:
            raise closed_error("deepgram", exc) from None

    async def audio(self) -> AsyncIterator[AudioChunk]:
        try:
            async for message in self._socket:
                if isinstance(message, bytes):
                    if not self._cancelled and message:
                        yield AudioChunk(message, self._output)
                    continue
                event = parse(message) or {}
                if event.get("type") in ("Flushed", "Cleared"):
                    return
        except ConnectionClosed as exc:
            if not self._cancelled and (exc.rcvd is None or exc.rcvd.code != 1000):
                raise closed_error("deepgram", exc) from None

    async def cancel(self) -> None:
        self._cancelled = True
        with contextlib.suppress(Exception):
            await send_json(self._socket, {"type": "Clear"})

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await send_json(self._socket, {"type": "Close"})
        with contextlib.suppress(Exception):
            await self._socket.close()

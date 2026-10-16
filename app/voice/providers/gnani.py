"""Gnani.ai (Vachana) streaming speech (checked against docs.gnani.ai and the
gnani-vachana SDK on 2026-10-16; docs/voice-interface.md).

Speech-to-text: wss://api.vachana.ai/stt/v3/stream. All settings are
upgrade headers, fixed for the connection: `x-api-key-id`, `lang_code`,
`x-sample-rate`, `x-format`, and the VAD headers. Audio goes up as binary
frames of exactly 1024 bytes (512 samples of 16-bit mono PCM) at real-time
pace. The server sends JSON "connected" (first; a bad key or header closes
the socket instead, so the adapter waits for it), "processing" (its VAD found
the end of speech), "transcript" (one final text per VAD segment) and
"error". The official SDK also accepts undocumented "speech_start" /
"vad_start" (read here as speech started) and "speech_end" / "vad_end".
There are no interim results and no flush message: to end a push-to-talk
turn, this adapter sends trailing silence (paced) so the server's VAD
closes the segment.

Text-to-speech: wss://api.vachana.ai/api/v1/tts, header `X-API-Key-ID`.
Each request is one whole text ({text, voice, model, language,
audio_config}); the server answers "start", base64 "audio" chunks and
"complete". Text is not streamed incrementally, so a reply is sent sentence
by sentence. The docs disagree on whether one socket takes several
requests; the adapter reuses it and reconnects if the server closed it.
Gnani's own clients also accept audio as binary frames, so this one does too.
There is no cancel message: barge-in closes the socket.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from app.voice.base import AudioChunk, AudioFormat, SpeechError, TranscriptEvent, TranscriptKind
from app.voice.providers._ws import closed_error, open_socket, parse, send_json
from app.voice.wav import WavStripper

STT_URL = "wss://api.vachana.ai/stt/v3/stream"
TTS_URL = "wss://api.vachana.ai/api/v1/tts"
FRAME_BYTES = 1024
STT_SAMPLE_RATES = (8_000, 16_000, 44_100, 48_000)
# Silence that ends a segment: above the server's min-silence (set below),
# then how long to wait for its "processing" before deciding nothing was said.
MIN_SILENCE_MS = 600
TRAILING_SILENCE_MS = 900
NO_SPEECH_WAIT_SECONDS = 2.0
CONNECTED_WAIT_SECONDS = 5.0
DEFAULT_TTS_MODEL = "timbre-v2.5"
TTS_SAMPLE_RATE = 24_000
# timbre-v2.5 voices listed by Gnani for each language; TTS_VOICE overrides.
DEFAULT_VOICES = {"en-IN": "Kaveri", "hi-IN": "Nalini"}


class GnaniSpeechToText:
    name = "gnani"
    model = "gnani-prisma-v2.5"  # no model header exists for streaming

    def __init__(self, api_key: str, *, model: str | None = None, url: str = STT_URL) -> None:
        self._key = api_key
        self._url = url

    async def open(self, audio: AudioFormat, language: str) -> _GnaniSttStream:
        if audio.sample_rate not in STT_SAMPLE_RATES:
            raise SpeechError(self.name, f"unsupported_sample_rate_{audio.sample_rate}")
        headers = {
            "x-api-key-id": self._key,
            "lang_code": language,
            "x-sample-rate": str(audio.sample_rate),
            "x-format": "transcribe",
            "x-min-silence-ms": str(MIN_SILENCE_MS),
        }
        socket = await open_socket(self.name, self._url, headers)
        try:
            first = parse(await asyncio.wait_for(socket.recv(), CONNECTED_WAIT_SECONDS))
        except TimeoutError:
            await socket.close()
            raise SpeechError(self.name, "not_connected", retryable=True) from None
        except ConnectionClosed as exc:  # 4001: authentication failed
            raise closed_error(self.name, exc) from None
        if first is None or first.get("type") != "connected":
            await socket.close()
            kind = first.get("type") if first else None
            raise SpeechError(self.name, f"not_connected:{kind}")
        return _GnaniSttStream(socket, audio)


class _GnaniSttStream:
    def __init__(self, socket: ClientConnection, audio: AudioFormat) -> None:
        self._socket = socket
        self._audio = audio
        self._buffer = bytearray()
        self._finished = False
        self._frame_seconds = FRAME_BYTES / audio.bytes_per_second()

    async def send(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        try:
            while len(self._buffer) >= FRAME_BYTES:
                frame = bytes(self._buffer[:FRAME_BYTES])
                del self._buffer[:FRAME_BYTES]
                await self._socket.send(frame)
        except ConnectionClosed as exc:
            raise closed_error("gnani", exc) from None

    async def finish(self) -> None:
        """Pad the last frame, then send paced silence so the server's VAD
        ends the segment."""
        if self._finished:
            return
        self._finished = True
        try:
            if self._buffer:
                await self._socket.send(bytes(self._buffer).ljust(FRAME_BYTES, b"\x00"))
                self._buffer.clear()
            silence = bytes(FRAME_BYTES)
            for _ in range(int(TRAILING_SILENCE_MS / 1000 / self._frame_seconds) + 1):
                await self._socket.send(silence)
                await asyncio.sleep(self._frame_seconds)
        except ConnectionClosed as exc:
            raise closed_error("gnani", exc) from None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        pending = 0  # segments the server is transcribing
        try:
            while True:
                if self._finished and pending == 0:
                    # Finished, nothing in progress: wait briefly for a late
                    # "processing", else there was nothing more to transcribe.
                    try:
                        message = await asyncio.wait_for(
                            self._socket.recv(), NO_SPEECH_WAIT_SECONDS
                        )
                    except TimeoutError:
                        return
                else:
                    message = await self._socket.recv()
                event = parse(message)
                if event is None:
                    continue
                kind = event.get("type")
                if kind == "processing":
                    pending += 1
                    yield TranscriptEvent(TranscriptKind.UTTERANCE_END)
                elif kind in ("speech_start", "vad_start"):
                    yield TranscriptEvent(TranscriptKind.SPEECH_STARTED)
                elif kind == "transcript":
                    pending = max(0, pending - 1)
                    yield TranscriptEvent(TranscriptKind.FINAL, str(event.get("text") or ""))
                    if self._finished and pending == 0:
                        return
                elif kind == "error":
                    raise SpeechError("gnani", str(event.get("message") or "error")[:80])
        except ConnectionClosed as exc:
            if exc.rcvd is None or exc.rcvd.code != 1000:
                raise closed_error("gnani", exc) from None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._socket.close()


class GnaniTextToSpeech:
    name = "gnani"
    output = AudioFormat(sample_rate=TTS_SAMPLE_RATE)

    def __init__(self, api_key: str, *, model: str | None = None, url: str = TTS_URL) -> None:
        self._key = api_key
        self.model = model or DEFAULT_TTS_MODEL
        self._url = url

    async def connect(self) -> ClientConnection:
        return await open_socket(self.name, self._url, {"X-API-Key-ID": self._key})

    async def open(self, voice: str | None, language: str) -> _GnaniTtsStream:
        chosen = voice or DEFAULT_VOICES.get(language) or DEFAULT_VOICES["en-IN"]
        return _GnaniTtsStream(self, await self.connect(), chosen, language)


class _GnaniTtsStream:
    def __init__(
        self, owner: GnaniTextToSpeech, socket: ClientConnection, voice: str, language: str
    ) -> None:
        self._owner = owner
        self._socket = socket
        self._voice = voice
        self._language = language
        self._texts: list[str] = []
        self._cancelled = False
        # The raw container has no headers; this guards against one anyway.
        self._wav = WavStripper("gnani", owner.output)

    async def send_text(self, text: str) -> None:
        if text.strip():
            self._texts.append(text.strip())

    async def flush(self) -> None:
        return None  # requests are sent one at a time from audio()

    def _request(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "voice": self._voice,
            "model": self._owner.model,
            "language": self._language,
            "audio_config": {
                "sample_rate": TTS_SAMPLE_RATE,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "raw",
            },
        }

    async def audio(self) -> AsyncIterator[AudioChunk]:
        for text in self._texts:
            if self._cancelled:
                return
            try:
                await send_json(self._socket, self._request(text))
            except ConnectionClosed:
                # The server closed after the previous request: reconnect once.
                self._socket = await self._owner.connect()
                await send_json(self._socket, self._request(text))
            try:
                async for message in self._socket:
                    if isinstance(message, bytes):
                        pcm = self._wav.feed(message) if not self._cancelled else b""
                        if pcm:
                            yield AudioChunk(pcm, self._owner.output)
                        continue
                    event = parse(message) or {}
                    kind = event.get("type")
                    if kind == "audio" and not self._cancelled:
                        data = event.get("data") or {}
                        pcm = self._wav.feed(base64.b64decode(str(data.get("audio") or "")))
                        if pcm:
                            yield AudioChunk(pcm, self._owner.output)
                    elif kind == "complete":
                        break
                    elif kind == "error":
                        raise SpeechError("gnani", str(event.get("message") or "error")[:80])
            except ConnectionClosed as exc:
                if self._cancelled:
                    return
                if exc.rcvd is None or exc.rcvd.code != 1000:
                    raise closed_error("gnani", exc) from None

    async def cancel(self) -> None:
        self._cancelled = True
        with contextlib.suppress(Exception):
            await self._socket.close()

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._socket.close()

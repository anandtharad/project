"""Sarvam AI streaming speech (checked against Sarvam's API reference,
guides and the sarvamai SDK on 2026-10-16; docs/voice-interface.md).

Speech-to-text: the realtime WebSocket
wss://api.sarvam.ai/speech-to-text-realtime/ws (labelled beta in one guide;
close code 4000 can mean the account is not enabled for it), header
`Api-Subscription-Key`, settings in the query (language_code required,
encoding linear16, sample_rate 8000 or 16000). Every message is JSON: audio
goes up base64 in {"event": "audio_input", "audio"}; push-to-talk uses
endpointing=manual with "speech_start" / "speech_end" (hands-free uses the
server's VAD); "end" closes gracefully. The server sends
"transcript.partial", "transcript.final" (one per utterance),
"vad.speech_start" / "vad.speech_end", "session.end" and "error".

Text-to-speech: wss://api.sarvam.ai/text-to-speech/ws?model=...; the first
message is {"type": "config", "data": {language_code, speaker, ...}}, text
goes in {"type": "text", "data": {"text"}}, {"type": "flush"} synthesises
it, audio comes back base64 in {"type": "audio", "data": {"audio"}}, and
{"type": "event", "data": {"event_type": "final"}} ends a reply. There is
no cancel message: barge-in closes the socket (Sarvam's documented way).

The docs disagree on the streaming TTS codecs: the reference says MP3 only,
the guides list linear16 and others, and the bulbul:v4-flash guide says the
WebSocket takes only wav and mp3. This adapter asks for wav, strips its
headers (checking the format against the one announced to the client), and
fails with a clear error on compressed audio instead of playing noise.
Realtime STT errors carry `is_fatal`; non-fatal ones (a rejected
config.update, which this adapter never sends) do not end the stream.
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from app.voice.base import AudioChunk, AudioFormat, SpeechError, TranscriptEvent, TranscriptKind
from app.voice.providers._ws import closed_error, open_socket, parse, send_json
from app.voice.wav import WavStripper

DEFAULT_BASE = "wss://api.sarvam.ai"
DEFAULT_STT_MODEL = "saaras:v3-realtime"
DEFAULT_TTS_MODEL = "bulbul:v4-flash"
# An example persona id from Sarvam's bulbul:v4-flash guide; set TTS_VOICE
# for another (v4-flash voices are persona ids, not v3 names).
DEFAULT_SPEAKER = "simran_en_customer"
TTS_SAMPLE_RATE = 24_000  # the streaming maximum
STT_SAMPLE_RATES = (8_000, 16_000)
MAX_TEXT_CHARACTERS = 500  # under the recommended size per text message


class SarvamSpeechToText:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        end_of_turn: Literal["client", "provider"] = "client",
        base: str = DEFAULT_BASE,
    ) -> None:
        self._key = api_key
        self.model = model or DEFAULT_STT_MODEL
        self._manual = end_of_turn == "client"
        self._base = base

    async def open(self, audio: AudioFormat, language: str) -> _SarvamSttStream:
        if audio.sample_rate not in STT_SAMPLE_RATES:
            raise SpeechError(self.name, f"unsupported_sample_rate_{audio.sample_rate}")
        query = urlencode(
            {
                "language_code": language,
                "model": self.model,
                "encoding": "linear16",
                "sample_rate": audio.sample_rate,
                "stream_type": "balanced",
                "endpointing": "manual" if self._manual else "vad",
            }
        )
        socket = await open_socket(
            self.name,
            f"{self._base}/speech-to-text-realtime/ws?{query}",
            {"Api-Subscription-Key": self._key},
        )
        return _SarvamSttStream(socket, self._manual)


class _SarvamSttStream:
    def __init__(self, socket: ClientConnection, manual: bool) -> None:
        self._socket = socket
        self._manual = manual
        self._started = False
        self._finished = False

    async def send(self, pcm: bytes) -> None:
        try:
            if self._manual and not self._started:
                self._started = True
                await send_json(self._socket, {"event": "speech_start"})
            await send_json(
                self._socket, {"event": "audio_input", "audio": base64.b64encode(pcm).decode()}
            )
        except ConnectionClosed as exc:
            raise closed_error("sarvam", exc) from None

    async def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self._manual and self._started:
                await send_json(self._socket, {"event": "speech_end"})
            await send_json(self._socket, {"event": "end"})
        except ConnectionClosed as exc:
            raise closed_error("sarvam", exc) from None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        try:
            async for message in self._socket:
                event = parse(message)
                if event is None:
                    continue
                kind = event.get("event")
                if kind == "transcript.final":
                    yield TranscriptEvent(TranscriptKind.FINAL, str(event.get("text") or ""))
                elif kind == "transcript.partial":
                    yield TranscriptEvent(TranscriptKind.INTERIM, str(event.get("text") or ""))
                elif kind == "vad.speech_start":
                    yield TranscriptEvent(TranscriptKind.SPEECH_STARTED)
                elif kind == "vad.speech_end":
                    yield TranscriptEvent(TranscriptKind.UTTERANCE_END)
                elif kind == "session.end":
                    return
                elif kind == "error" and event.get("is_fatal") is not False:
                    raise SpeechError("sarvam", str(event.get("code") or "error"))
        except ConnectionClosed as exc:
            if exc.rcvd is None or exc.rcvd.code != 1000:
                raise closed_error("sarvam", exc) from None

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._socket.close()


class SarvamTextToSpeech:
    name = "sarvam"
    output = AudioFormat(sample_rate=TTS_SAMPLE_RATE)

    def __init__(self, api_key: str, *, model: str | None = None, base: str = DEFAULT_BASE) -> None:
        self._key = api_key
        self.model = model or DEFAULT_TTS_MODEL
        self._base = base

    async def open(self, voice: str | None, language: str) -> _SarvamTtsStream:
        query = urlencode({"model": self.model, "send_completion_event": "true"})
        socket = await open_socket(
            self.name,
            f"{self._base}/text-to-speech/ws?{query}",
            {"Api-Subscription-Key": self._key},
        )
        try:
            await send_json(
                socket,
                {
                    "type": "config",
                    "data": {
                        "language_code": language,
                        "speaker": voice or DEFAULT_SPEAKER,
                        "speech_sample_rate": TTS_SAMPLE_RATE,
                        "output_audio_codec": "wav",
                    },
                },
            )
        except ConnectionClosed as exc:
            raise closed_error(self.name, exc) from None
        return _SarvamTtsStream(socket, self.output)


class _SarvamTtsStream:
    def __init__(self, socket: ClientConnection, output: AudioFormat) -> None:
        self._socket = socket
        self._output = output
        self._cancelled = False
        self._wav = WavStripper("sarvam", output)

    async def send_text(self, text: str) -> None:
        try:
            for start in range(0, len(text), MAX_TEXT_CHARACTERS):
                piece = text[start : start + MAX_TEXT_CHARACTERS]
                if piece.strip():
                    await send_json(self._socket, {"type": "text", "data": {"text": piece}})
        except ConnectionClosed as exc:
            raise closed_error("sarvam", exc) from None

    async def flush(self) -> None:
        try:
            await send_json(self._socket, {"type": "flush"})
        except ConnectionClosed as exc:
            raise closed_error("sarvam", exc) from None

    async def audio(self) -> AsyncIterator[AudioChunk]:
        try:
            async for message in self._socket:
                event = parse(message) or {}
                kind = event.get("type")
                data = event.get("data") or {}
                if kind == "audio" and not self._cancelled:
                    pcm = self._wav.feed(
                        pcm_from(
                            base64.b64decode(str(data.get("audio") or "")),
                            str(data.get("content_type") or ""),
                        )
                    )
                    if pcm:
                        yield AudioChunk(pcm, self._output)
                elif kind == "event" and data.get("event_type") == "final":
                    return
                elif kind == "error":
                    raise SpeechError("sarvam", str(data.get("code") or "error"))
        except ConnectionClosed as exc:
            if not self._cancelled and (exc.rcvd is None or exc.rcvd.code != 1000):
                raise closed_error("sarvam", exc) from None

    async def cancel(self) -> None:
        # No cancel message exists; closing the socket stops synthesis.
        self._cancelled = True
        with contextlib.suppress(Exception):
            await self._socket.close()

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._socket.close()


_COMPRESSED = ("mp3", "mpeg", "aac", "opus", "ogg", "flac")


def pcm_from(audio: bytes, content_type: str) -> bytes:
    """The chunk, unless it is compressed audio (MP3, ...), which is refused
    rather than played as noise. WAV headers are removed by WavStripper."""
    if any(name in content_type.lower() for name in _COMPRESSED):
        raise SpeechError("sarvam", f"compressed_audio_returned:{content_type}")
    return audio

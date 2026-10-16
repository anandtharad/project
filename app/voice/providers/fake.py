"""Speech providers for tests and for trying the voice client without keys.

- `FakeSpeechToText` returns scripted transcripts, one per utterance; with
  none scripted it describes the audio it got ("(3.2 seconds of audio)").
- `FakeTextToSpeech` "speaks" a short tone per word, so playback can be
  heard in the browser client. It honours `cancel()`.

Neither recognises or synthesises speech.
"""

from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.voice.base import (
    AudioChunk,
    AudioFormat,
    SpeechError,
    TranscriptEvent,
    TranscriptKind,
)


@dataclass
class FakeSpeechToText:
    transcripts: list[str] = field(default_factory=list)
    fail: bool = False
    name: str = "fake"
    model: str = "fake-stt"
    opened: int = 0

    async def open(self, audio: AudioFormat, language: str) -> _FakeSttStream:
        if self.fail:
            raise SpeechError(self.name, "unavailable", retryable=True)
        self.opened += 1
        text = self.transcripts.pop(0) if self.transcripts else None
        return _FakeSttStream(audio, text)


class _FakeSttStream:
    def __init__(self, audio: AudioFormat, text: str | None) -> None:
        self._audio = audio
        self._text = text
        self._bytes = 0
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._interim_sent = False

    async def send(self, pcm: bytes) -> None:
        self._bytes += len(pcm)
        if not self._interim_sent:
            self._interim_sent = True
            await self._queue.put(TranscriptEvent(TranscriptKind.SPEECH_STARTED))
            if self._text:
                await self._queue.put(
                    TranscriptEvent(TranscriptKind.INTERIM, self._text.split()[0])
                )

    async def finish(self) -> None:
        seconds = self._bytes / self._audio.bytes_per_second()
        text = self._text if self._text is not None else f"({seconds:.1f} seconds of audio)"
        if text:
            await self._queue.put(TranscriptEvent(TranscriptKind.FINAL, text, 1.0))
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while (event := await self._queue.get()) is not None:
            yield event

    async def close(self) -> None:
        await self._queue.put(None)


@dataclass
class FakeTextToSpeech:
    output: AudioFormat = field(default_factory=lambda: AudioFormat(sample_rate=16000))
    fail: bool = False
    name: str = "fake"
    model: str = "fake-tts"
    seconds_per_word: float = 0.12
    spoken: list[str] = field(default_factory=list)
    cancelled: int = 0

    async def open(self, voice: str | None, language: str) -> _FakeTtsStream:
        if self.fail:
            raise SpeechError(self.name, "unavailable", retryable=True)
        return _FakeTtsStream(self)


class _FakeTtsStream:
    def __init__(self, owner: FakeTextToSpeech) -> None:
        self._owner = owner
        self._text: list[str] = []
        self._flushed = False
        self._cancelled = False

    async def send_text(self, text: str) -> None:
        self._text.append(text)

    async def flush(self) -> None:
        self._flushed = True
        self._owner.spoken.append("".join(self._text).strip())

    async def audio(self) -> AsyncIterator[AudioChunk]:
        fmt = self._owner.output
        for index, _ in enumerate("".join(self._text).split()):
            if self._cancelled or not self._flushed:
                return
            yield AudioChunk(_tone(fmt, self._owner.seconds_per_word, 330 + 40 * (index % 4)), fmt)
            await asyncio.sleep(0)

    async def cancel(self) -> None:
        self._cancelled = True
        self._owner.cancelled += 1

    async def close(self) -> None:
        return None


def _tone(fmt: AudioFormat, seconds: float, hertz: float) -> bytes:
    count = int(fmt.sample_rate * seconds)
    fade = max(1, count // 10)
    samples = []
    for i in range(count):
        envelope = min(1.0, i / fade, (count - i) / fade)
        samples.append(int(6000 * envelope * math.sin(2 * math.pi * hertz * i / fmt.sample_rate)))
    return struct.pack(f"<{count}h", *samples)

"""Provider-neutral streaming speech interfaces.

Speech-to-text: open a stream, send audio chunks as the person speaks, and
read transcript events (interim and final text, speech start, utterance
end). `finish()` says the person stopped (push-to-talk released); the
provider then flushes its last final transcript.

Text-to-speech: open a stream, send text, `flush()` to synthesise what was
sent, and read audio chunks. `cancel()` stops synthesis and drops audio not
yet delivered: half duplex never needs it, but full duplex with barge-in
does (the person starts talking over the reply), so every provider
implements it now.

Audio inside the application is always raw 16-bit little-endian PCM, mono,
at a declared sample rate; providers convert at their edge.
"""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AudioFormat:
    """Raw PCM: 16-bit signed little-endian samples, mono."""

    sample_rate: int
    encoding: str = "linear16"
    channels: int = 1

    def bytes_per_second(self) -> int:
        return self.sample_rate * 2 * self.channels


class TranscriptKind(enum.StrEnum):
    INTERIM = "interim"  # may still change
    FINAL = "final"  # this segment will not change
    SPEECH_STARTED = "speech_started"
    UTTERANCE_END = "utterance_end"  # the provider detected the end of the person's turn


@dataclass(frozen=True)
class TranscriptEvent:
    kind: TranscriptKind
    text: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    format: AudioFormat


class SpeechError(Exception):
    """A provider failed; `retryable` when a new attempt may succeed."""

    def __init__(self, provider: str, code: str, *, retryable: bool = False) -> None:
        super().__init__(f"{provider}: {code}")
        self.provider = provider
        self.code = code
        self.retryable = retryable


class SpeechToTextStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...

    async def finish(self) -> None:
        """No more audio for this utterance: flush the final transcript."""
        ...

    def events(self) -> AsyncIterator[TranscriptEvent]:
        """Transcript events. The iterator ends after `finish()`, once the
        last final transcript is delivered, or when the stream closes."""
        ...

    async def close(self) -> None: ...


class SpeechToText(Protocol):
    name: str
    model: str

    async def open(self, audio: AudioFormat, language: str) -> SpeechToTextStream: ...


class TextToSpeechStream(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def flush(self) -> None:
        """Synthesise everything sent so far."""
        ...

    def audio(self) -> AsyncIterator[AudioChunk]:
        """Audio until the flushed text is spoken (then the iterator ends)."""
        ...

    async def cancel(self) -> None:
        """Stop now and drop undelivered audio (barge-in)."""
        ...

    async def close(self) -> None: ...


class TextToSpeech(Protocol):
    name: str
    model: str
    output: AudioFormat

    async def open(self, voice: str | None, language: str) -> TextToSpeechStream: ...

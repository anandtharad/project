"""One voice session: the person speaks or types, the companion replies in
text, audio, or both.

States:

    LISTENING --audio/text--> (TRANSCRIBING) --> THINKING --> SPEAKING --> LISTENING
                                                     \\-- text output only --/

- **Half duplex (now).** One side at a time. Audio that arrives while the
  companion is thinking or speaking is dropped and counted, and a typed
  message is refused as busy. The client is told the state, so it can mute
  the microphone.
- **Full duplex (later).** The same machine, but speech while SPEAKING is a
  barge-in: the reply's audio is cancelled (`TextToSpeechStream.cancel`)
  and the session goes back to LISTENING with the new audio. The path is
  implemented and tested here; configuration only allows "half" until echo
  cancellation and the partial-reply bookkeeping are designed
  (docs/voice-interface.md).

A spoken turn is an ordinary turn: the final transcript goes through the
same pipeline as typed text (safety, memory, knowledge, output checks), and
only the validated reply is spoken, with citation markers removed.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.observability import log_event
from app.rag.evidence import strip_markers
from app.voice.base import (
    AudioFormat,
    SpeechError,
    SpeechToText,
    SpeechToTextStream,
    TextToSpeech,
    TranscriptKind,
)

logger = logging.getLogger("app.voice")

Duplex = Literal["half", "full"]


class State(enum.StrEnum):
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"  # the person finished; waiting for the final transcript
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSED = "closed"


@dataclass(frozen=True)
class TurnReply:
    text: str
    user_message_id: str
    assistant_message_id: str
    extra: dict[str, Any] = field(default_factory=dict)  # safety notice, sources


@dataclass(frozen=True)
class VoiceConfig:
    language: str
    input_audio: AudioFormat
    duplex: Duplex = "half"
    # Who decides the person has finished: "client" (push-to-talk: the
    # release, so a pause while holding never cuts them off) or "provider"
    # (hands-free: the speech-to-text provider's end-of-utterance).
    end_of_turn: Literal["client", "provider"] = "client"
    voice: str | None = None
    max_utterance_seconds: float = 60.0
    final_transcript_timeout_seconds: float = 5.0
    # After the last audio chunk, how long to wait for the client's
    # "playback_done" beyond the audio's own duration.
    playback_grace_seconds: float = 5.0


# What the transport sends to the client: a JSON event or binary audio.
Send = Callable[[dict[str, Any]], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]
RunTurn = Callable[[str, str], Awaitable[TurnReply]]  # (text, modality) -> reply

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_FORMATTING = re.compile(r"[*_`#>]+")


def speakable(text: str) -> str:
    """The reply as it should be spoken: no citation markers or formatting."""
    text = _FORMATTING.sub("", strip_markers(text))
    # Separators left by removed markers ("helps [S1], [S2].") are not spoken.
    text = re.sub(r"[\s,;]+(?=[.!?])", "", text)
    text = re.sub(r"\s+([,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s.strip()]


class VoiceSession:
    def __init__(
        self,
        *,
        config: VoiceConfig,
        stt: SpeechToText | None,
        tts: TextToSpeech | None,
        run_turn: RunTurn,
        send: Send,
        send_audio: SendAudio,
        audio_output: bool,
    ) -> None:
        self.config = config
        self._stt = stt
        self._tts = tts
        self._run_turn = run_turn
        self._send = send
        self._send_audio = send_audio
        self.audio_output = audio_output and tts is not None
        self.state = State.LISTENING
        self.dropped_audio_bytes = 0
        self._stream: SpeechToTextStream | None = None
        self._reader: asyncio.Task[None] | None = None
        self._finals: list[str] = []
        self._utterance_bytes = 0
        self._work: asyncio.Task[None] | None = None
        self._speaking: Any = None  # the open TextToSpeechStream while SPEAKING
        self._playback_done = asyncio.Event()
        # Incremented by a barge-in, so an interrupted turn's task does not
        # change the state of the turn that replaced it.
        self._turn = 0
        # Speech recognition failed for this utterance: drop its audio (one
        # error, no reconnect per frame) until the person stops talking.
        self._stt_failed = False

    # --- inbound events -------------------------------------------------------

    async def start(self) -> None:
        await self._send(
            {
                "type": "ready",
                "duplex": self.config.duplex,
                "input": ["text", *(["audio"] if self._stt else [])],
                "output": ["text", *(["audio"] if self.audio_output else [])],
                "input_audio": _format(self.config.input_audio),
                "output_audio": _format(self._tts.output)
                if self.audio_output and self._tts
                else None,
                "language": self.config.language,
            }
        )
        await self._set_state(State.LISTENING)

    async def on_audio(self, pcm: bytes) -> None:
        if self.state == State.SPEAKING and self.config.duplex == "full":
            await self._barge_in()
        if self.state != State.LISTENING or self._stt is None or self._stt_failed:
            self.dropped_audio_bytes += len(pcm)
            return
        if self._stream is None:
            try:
                await self._open_stt()
            except SpeechError as exc:
                self._stt_failed = True
                self.dropped_audio_bytes += len(pcm)
                log_event(logger, "voice.stt_failed", level=logging.WARNING, code=exc.code)
                await self._send(_STT_ERROR)
                return
        assert self._stream is not None
        self._utterance_bytes += len(pcm)
        await self._stream.send(pcm)
        limit = self.config.max_utterance_seconds * self.config.input_audio.bytes_per_second()
        if self._utterance_bytes >= limit:
            await self.on_end_of_speech()

    async def on_end_of_speech(self) -> None:
        """The person stopped talking (push-to-talk released, or the
        provider's end-of-utterance)."""
        if self._stt_failed:
            self._stt_failed = False
            return
        if self.state != State.LISTENING or self._stream is None:
            return
        await self._set_state(State.TRANSCRIBING)
        self._work = asyncio.create_task(self._finish_utterance())

    async def on_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.state != State.LISTENING or self._stream is not None:
            await self._send({"type": "error", "code": "busy", "message": "Wait for the reply."})
            return
        await self._set_state(State.THINKING)
        self._work = asyncio.create_task(self._respond(text, "text"))

    async def notify(self, event: dict[str, Any]) -> None:
        """Send an event to the client (the transport's own errors)."""
        await self._send(event)

    async def on_playback_done(self) -> None:
        self._playback_done.set()

    async def close(self) -> None:
        self.state = State.CLOSED
        for task in (self._work, self._reader):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self._speaking is not None:
            with contextlib.suppress(Exception):
                await self._speaking.cancel()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.close()
        log_event(logger, "voice.closed", dropped_audio_bytes=self.dropped_audio_bytes)

    async def wait_idle(self) -> None:
        """Tests and the transport: wait for the current turn to finish."""
        if self._work is not None:
            await asyncio.gather(self._work, return_exceptions=True)

    # --- speech-to-text ---------------------------------------------------------

    async def _open_stt(self) -> None:
        assert self._stt is not None
        self._finals = []
        self._utterance_bytes = 0
        self._stream = await self._stt.open(self.config.input_audio, self.config.language)
        self._reader = asyncio.create_task(self._read_transcripts(self._stream))

    async def _read_transcripts(self, stream: SpeechToTextStream) -> None:
        async for event in stream.events():
            if event.kind == TranscriptKind.FINAL and event.text.strip():
                self._finals.append(event.text.strip())
                await self._send({"type": "transcript", "final": True, "text": event.text})
            elif event.kind == TranscriptKind.INTERIM and event.text.strip():
                await self._send({"type": "transcript", "final": False, "text": event.text})
            elif (
                event.kind == TranscriptKind.UTTERANCE_END
                and self.config.end_of_turn == "provider"
                and self.state == State.LISTENING
            ):
                await self.on_end_of_speech()

    async def _finish_utterance(self) -> None:
        stream, reader = self._stream, self._reader
        assert stream is not None and reader is not None
        try:
            await stream.finish()
            await asyncio.wait_for(
                asyncio.shield(reader), self.config.final_transcript_timeout_seconds
            )
        except (TimeoutError, SpeechError) as exc:
            log_event(logger, "voice.stt_failed", level=logging.WARNING, error=type(exc).__name__)
            await self._send(_STT_ERROR)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            with contextlib.suppress(Exception):
                await stream.close()
            self._stream, self._reader = None, None
        transcript = " ".join(self._finals).strip()
        if not transcript:
            await self._send({"type": "no_speech"})
            await self._set_state(State.LISTENING)
            return
        await self._set_state(State.THINKING)
        await self._respond(transcript, "voice")

    # --- the turn and text-to-speech ---------------------------------------------

    async def _respond(self, text: str, modality: str) -> None:
        started = time.monotonic()
        turn = self._turn
        try:
            reply = await self._run_turn(text, modality)
        except Exception as exc:
            code = getattr(exc, "code", "turn_failed")
            await self._send(
                {
                    "type": "error",
                    "code": str(code),
                    "message": str(getattr(exc, "message", "The reply failed.")),
                }
            )
            await self._set_state(State.LISTENING)
            return
        await self._send(
            {
                "type": "reply",
                "text": reply.text,
                "user_message_id": reply.user_message_id,
                "assistant_message_id": reply.assistant_message_id,
                **reply.extra,
            }
        )
        if self.audio_output and turn == self._turn:
            await self._speak(reply.text, turn)
        log_event(
            logger,
            "voice.turn",
            modality=modality,
            audio_output=self.audio_output,
            total_ms=round((time.monotonic() - started) * 1000),
        )
        if turn == self._turn:
            await self._set_state(State.LISTENING)

    async def _speak(self, text: str, turn: int) -> None:
        assert self._tts is not None
        spoken = speakable(text)
        if not spoken:
            return
        await self._set_state(State.SPEAKING)
        self._playback_done.clear()
        seconds = 0.0
        try:
            stream = await self._tts.open(self.config.voice, self.config.language)
            self._speaking = stream
            try:
                for sentence in sentences(spoken):
                    await stream.send_text(sentence + " ")
                await stream.flush()
                async for chunk in stream.audio():
                    if turn != self._turn:
                        break  # cancelled by a barge-in
                    seconds += len(chunk.pcm) / chunk.format.bytes_per_second()
                    await self._send_audio(chunk.pcm)
            finally:
                self._speaking = None
                with contextlib.suppress(Exception):
                    await stream.close()
        except SpeechError as exc:
            log_event(logger, "voice.tts_failed", level=logging.WARNING, code=exc.code)
            await self._send(
                {
                    "type": "error",
                    "code": "tts_unavailable",
                    "message": "The reply could not be spoken; it is shown as text.",
                }
            )
            return
        if turn != self._turn:
            return
        await self._send({"type": "audio_end", "seconds": round(seconds, 2)})
        # Half duplex: listen again only once the client has played it all.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._playback_done.wait(), seconds + self.config.playback_grace_seconds
            )

    async def _barge_in(self) -> None:
        """Full duplex only: the person talks over the reply."""
        self._turn += 1
        if self._speaking is not None:
            with contextlib.suppress(Exception):
                await self._speaking.cancel()
        await self._send({"type": "interrupted"})
        await self._set_state(State.LISTENING)

    async def _set_state(self, state: State) -> None:
        if self.state == State.CLOSED:
            return
        self.state = state
        await self._send({"type": "state", "state": state.value})


_STT_ERROR = {
    "type": "error",
    "code": "stt_unavailable",
    "message": "Speech recognition failed; you can type instead.",
}


def _format(audio: AudioFormat) -> dict[str, Any]:
    return {
        "encoding": audio.encoding,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
    }

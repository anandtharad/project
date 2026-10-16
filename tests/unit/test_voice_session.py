"""The voice session state machine (app/voice/session.py) with fake
providers: both input modes, both output modes, half duplex, failures, and
the full-duplex barge-in path."""

from __future__ import annotations

import asyncio
from typing import Any

from app.voice.base import AudioFormat, TranscriptEvent, TranscriptKind
from app.voice.providers.fake import FakeSpeechToText, FakeTextToSpeech
from app.voice.session import State, TurnReply, VoiceConfig, VoiceSession, sentences, speakable

PCM = b"\x00\x01" * 1600  # 0.1 s at 16 kHz


class Harness:
    def __init__(
        self,
        *,
        transcripts: list[str] | None = None,
        reply: str = "That sounds like a long week [S1]. What happened?",
        audio_output: bool = True,
        duplex: str = "half",
        stt: FakeSpeechToText | None = None,
        tts: FakeTextToSpeech | None = None,
        fail_turn: Exception | None = None,
        max_utterance_seconds: float = 60.0,
        end_of_turn: str = "client",
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.turns: list[tuple[str, str]] = []
        self.reply = reply
        self.fail_turn = fail_turn
        self.gate: asyncio.Event | None = None
        self.thinking = asyncio.Event()
        self.stt = stt if stt is not None else FakeSpeechToText(list(transcripts or []))
        self.tts = tts if tts is not None else FakeTextToSpeech()
        self.session = VoiceSession(
            config=VoiceConfig(
                language="en-IN",
                input_audio=AudioFormat(sample_rate=16000),
                duplex=duplex,  # type: ignore[arg-type]
                playback_grace_seconds=0.2,
                max_utterance_seconds=max_utterance_seconds,
                end_of_turn=end_of_turn,  # type: ignore[arg-type]
            ),
            stt=self.stt,
            tts=self.tts,
            run_turn=self.run_turn,
            send=self.send,
            send_audio=self.send_audio,
            audio_output=audio_output,
        )

    async def run_turn(self, text: str, modality: str) -> TurnReply:
        self.turns.append((text, modality))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_turn is not None:
            raise self.fail_turn
        return TurnReply(self.reply, "u1", "a1", {"sources": []})

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if event.get("state") == "thinking":
            self.thinking.set()
        if event.get("type") == "audio_end":
            # The client finishes playing straight away.
            asyncio.get_running_loop().call_soon(
                lambda: asyncio.ensure_future(self.session.on_playback_done())
            )

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    def states(self) -> list[str]:
        return [e["state"] for e in self.events if e["type"] == "state"]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]


async def test_a_spoken_turn_is_transcribed_answered_and_spoken() -> None:
    h = Harness(transcripts=["work has been a lot this week"])
    await h.session.start()
    assert h.of("ready")[0]["input"] == ["text", "audio"]
    assert h.of("ready")[0]["output"] == ["text", "audio"]
    for _ in range(3):
        await h.session.on_audio(PCM)
    await h.session.on_end_of_speech()
    await h.session.wait_idle()
    assert h.turns == [("work has been a lot this week", "voice")]
    assert [e["final"] for e in h.of("transcript")] == [False, True]
    assert h.of("reply")[0]["text"] == "That sounds like a long week [S1]. What happened?"
    # Spoken without the citation marker; audio sent; listening again after playback.
    assert h.tts.spoken == ["That sounds like a long week. What happened?"]
    assert h.audio and h.of("audio_end")
    assert h.states() == ["listening", "transcribing", "thinking", "speaking", "listening"]
    assert h.session.state == State.LISTENING


async def test_half_duplex_drops_audio_and_refuses_text_while_busy() -> None:
    h = Harness(transcripts=["hello there"])
    h.gate = asyncio.Event()
    await h.session.start()
    await h.session.on_audio(PCM)
    await h.session.on_end_of_speech()
    await h.thinking.wait()
    await h.session.on_audio(PCM)  # the person talks while the companion thinks
    await h.session.on_text("are you there?")
    assert h.session.dropped_audio_bytes == len(PCM)
    assert h.of("error")[-1]["code"] == "busy"
    h.gate.set()
    await h.session.wait_idle()
    assert h.turns == [("hello there", "voice")]  # the typed message was not taken


async def test_typed_input_with_text_only_or_audio_output() -> None:
    spoken = Harness()
    await spoken.session.start()
    await spoken.session.on_text("how are you")
    await spoken.session.wait_idle()
    assert spoken.turns == [("how are you", "text")] and spoken.audio

    text_only = Harness(audio_output=False)
    await text_only.session.start()
    assert text_only.of("ready")[0]["output"] == ["text"]
    await text_only.session.on_text("how are you")
    await text_only.session.wait_idle()
    assert text_only.of("reply") and not text_only.audio
    assert text_only.states() == ["listening", "thinking", "listening"]


async def test_no_speech_and_stt_failure_return_to_listening() -> None:
    silent = Harness(transcripts=[""])
    await silent.session.start()
    await silent.session.on_audio(PCM)
    await silent.session.on_end_of_speech()
    await silent.session.wait_idle()
    assert silent.of("no_speech") and silent.turns == []
    assert silent.session.state == State.LISTENING

    broken = Harness(stt=FakeSpeechToText(fail=True))
    await broken.session.start()
    for _ in range(3):
        await broken.session.on_audio(PCM)
    # One error, no reconnect attempt per frame; the audio is dropped.
    assert [e["code"] for e in broken.of("error")] == ["stt_unavailable"]
    assert broken.session.dropped_audio_bytes == 3 * len(PCM)
    await broken.session.on_end_of_speech()
    assert broken.session.state == State.LISTENING
    # Typing still works when speech recognition does not.
    await broken.session.on_text("typing instead")
    await broken.session.wait_idle()
    assert broken.turns == [("typing instead", "text")]


async def test_tts_and_turn_failures_are_reported_and_the_session_continues() -> None:
    h = Harness(tts=FakeTextToSpeech(fail=True))
    await h.session.start()
    await h.session.on_text("hi")
    await h.session.wait_idle()
    assert h.of("reply") and h.of("error")[-1]["code"] == "tts_unavailable"
    assert h.session.state == State.LISTENING

    class Unavailable(Exception):
        code = "llm_unavailable"
        message = "The model is unavailable."

    failing = Harness(fail_turn=Unavailable())
    await failing.session.start()
    await failing.session.on_text("hi")
    await failing.session.wait_idle()
    assert failing.of("error")[-1]["code"] == "llm_unavailable"
    assert failing.session.state == State.LISTENING and not failing.of("reply")


async def test_provider_end_of_utterance_and_the_length_limit_end_the_turn() -> None:
    class EndpointingStt(FakeSpeechToText):
        async def open(self, audio: AudioFormat, language: str) -> Any:
            stream = await super().open(audio, language)
            await stream._queue.put(TranscriptEvent(TranscriptKind.UTTERANCE_END))
            return stream

    # Push-to-talk ignores the provider's end of utterance (a pause while
    # holding the button must not end the turn).
    held = Harness(stt=EndpointingStt(["not yet"]))
    await held.session.start()
    await held.session.on_audio(PCM)
    await asyncio.sleep(0.05)
    assert held.turns == [] and held.session.state == State.LISTENING
    await held.session.on_end_of_speech()
    await held.session.wait_idle()
    assert held.turns == [("not yet", "voice")]

    h = Harness(stt=EndpointingStt(["said by the provider"]), end_of_turn="provider")
    await h.session.start()
    await h.session.on_audio(PCM)
    for _ in range(50):
        if h.turns:
            break
        await asyncio.sleep(0.01)
    await h.session.wait_idle()
    assert h.turns == [("said by the provider", "voice")]

    capped = Harness(transcripts=["a long one"], max_utterance_seconds=0.2)
    await capped.session.start()
    await capped.session.on_audio(PCM)
    await capped.session.on_audio(PCM)  # 0.2 s reached: the turn ends by itself
    await capped.session.wait_idle()
    assert capped.turns == [("a long one", "voice")]


async def test_full_duplex_barge_in_cancels_the_reply() -> None:
    tts = FakeTextToSpeech(seconds_per_word=0.01)
    h = Harness(transcripts=["first", "second thing"], duplex="full", tts=tts)
    h.reply = " ".join(["word"] * 400)
    gate = asyncio.Event()
    original = h.send_audio

    async def slow_audio(pcm: bytes) -> None:
        await original(pcm)
        if len(h.audio) == 3:
            gate.set()
        await asyncio.sleep(0.001)

    h.session._send_audio = slow_audio
    await h.session.start()
    await h.session.on_audio(PCM)
    await h.session.on_end_of_speech()
    await gate.wait()
    assert h.session.state == State.SPEAKING
    await h.session.on_audio(PCM)  # the person talks over the reply
    assert h.of("interrupted") and tts.cancelled == 1
    assert h.session.state == State.LISTENING
    await h.session.on_end_of_speech()
    await h.session.wait_idle()
    assert h.turns == [("first", "voice"), ("second thing", "voice")]
    assert len(h.audio) < 400 + 400  # the first reply was cut short


def test_speakable_text() -> None:
    assert speakable("**Sleep** helps [S1], [S2]. Try `this`.") == "Sleep helps. Try this."
    assert sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]

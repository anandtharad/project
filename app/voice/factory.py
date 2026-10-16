"""Speech providers from settings (STT_PROVIDER, TTS_PROVIDER)."""

from __future__ import annotations

from app.config import Settings
from app.voice.base import AudioFormat, SpeechToText, TextToSpeech
from app.voice.providers.fake import FakeSpeechToText, FakeTextToSpeech


def _key(settings: Settings, provider: str) -> str:
    secret = getattr(settings, f"{provider}_api_key")
    if secret is None:  # Settings validation already requires it
        raise ValueError(f"{provider.upper()}_API_KEY is not set")
    return str(secret.get_secret_value())


def build_stt(settings: Settings) -> SpeechToText | None:
    provider = settings.stt_provider
    if not settings.voice_enabled or provider == "none":
        return None
    if provider == "fake":
        return FakeSpeechToText()
    if provider == "deepgram":
        from app.voice.providers.deepgram import DeepgramSpeechToText

        return DeepgramSpeechToText(_key(settings, provider), model=settings.stt_model)
    if provider == "sarvam":
        from app.voice.providers.sarvam import SarvamSpeechToText

        return SarvamSpeechToText(
            _key(settings, provider),
            model=settings.stt_model,
            end_of_turn=settings.voice_end_of_turn,
        )
    from app.voice.providers.gnani import GnaniSpeechToText

    return GnaniSpeechToText(_key(settings, provider), model=settings.stt_model)


def build_tts(settings: Settings) -> TextToSpeech | None:
    provider = settings.tts_provider
    if not settings.voice_enabled or provider == "none":
        return None
    if provider == "fake":
        return FakeTextToSpeech(output=AudioFormat(sample_rate=16_000))
    if provider == "deepgram":
        from app.voice.providers.deepgram import DeepgramTextToSpeech

        return DeepgramTextToSpeech(_key(settings, provider), model=settings.tts_model)
    if provider == "sarvam":
        from app.voice.providers.sarvam import SarvamTextToSpeech

        return SarvamTextToSpeech(_key(settings, provider), model=settings.tts_model)
    from app.voice.providers.gnani import GnaniTextToSpeech

    return GnaniTextToSpeech(_key(settings, provider), model=settings.tts_model)

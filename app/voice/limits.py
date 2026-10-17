"""Per-user limits for voice (docs/voice-interface.md, "Limits").

Every voice session holds a speech-to-text connection on the operator's
provider account, and recognised audio is billed whether or not it becomes a
message. So one person can hold only a few sessions at a time, and can send
only so many seconds of audio a day. (Pacing faster than real time is
refused per session, in VoiceSession.)

Like the tickets, this lives in the process's memory: with several workers,
each enforces its own share.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime


def _today() -> date:
    return datetime.now(UTC).date()


class VoiceLimits:
    def __init__(
        self,
        *,
        max_sessions_per_user: int,
        audio_seconds_per_day: float,
        today: Callable[[], date] = _today,
    ) -> None:
        self._max_sessions = max_sessions_per_user
        self._audio_budget = audio_seconds_per_day
        self._today = today
        self._sessions: Counter[uuid.UUID] = Counter()
        self._audio: dict[uuid.UUID, tuple[date, float]] = {}

    def can_open(self, user_id: uuid.UUID) -> bool:
        return self._sessions[user_id] < self._max_sessions

    def opened(self, user_id: uuid.UUID) -> None:
        self._sessions[user_id] += 1

    def closed(self, user_id: uuid.UUID) -> None:
        self._sessions[user_id] -= 1
        if self._sessions[user_id] <= 0:
            del self._sessions[user_id]

    def active(self, user_id: uuid.UUID) -> int:
        return self._sessions[user_id]

    def charge_audio(self, user_id: uuid.UUID, seconds: float) -> bool:
        """Count this audio against today's budget; False once it is spent
        (the audio is then not sent to the provider)."""
        today = self._today()
        day, used = self._audio.get(user_id, (today, 0.0))
        if day != today:
            used = 0.0
        if used + seconds > self._audio_budget:
            self._audio[user_id] = (today, used)
            return False
        self._audio[user_id] = (today, used + seconds)
        return True

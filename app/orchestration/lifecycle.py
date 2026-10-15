"""Turn lifecycle stages and per-turn timing."""

from __future__ import annotations

import time
from enum import StrEnum


class TurnStage(StrEnum):
    RECEIVED = "received"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    USER_MESSAGE_PERSISTED = "user_message_persisted"
    CONTEXT_LOADED = "context_loaded"
    SAFETY_ASSESSED = "safety_assessed"
    STATE_UPDATED = "state_updated"
    MEMORIES_RETRIEVED = "memories_retrieved"
    DECISIONS_RESOLVED = "decisions_resolved"
    POLICY_CREATED = "policy_created"
    CONTEXT_BUILT = "context_built"
    RESPONSE_GENERATED = "response_generated"
    RESPONSE_VALIDATED = "response_validated"
    SAFETY_CHECKED = "safety_checked"
    ASSISTANT_MESSAGE_PERSISTED = "assistant_message_persisted"
    COMPLETED = "completed"


class TurnTrace:
    """Records when each stage was reached, relative to turn start."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self._start = time.perf_counter()
        self._marks: dict[TurnStage, float] = {}
        self.mark(TurnStage.RECEIVED)

    def mark(self, stage: TurnStage) -> None:
        self._marks[stage] = (time.perf_counter() - self._start) * 1000

    @property
    def last_stage(self) -> TurnStage:
        return next(reversed(self._marks))

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start

    def timings_ms(self) -> dict[str, float]:
        return {stage.value: round(ms, 2) for stage, ms in self._marks.items()}

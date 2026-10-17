"""Per-user voice limits, the ticket cap and the provider-logging cap."""

from __future__ import annotations

import logging
import uuid
from datetime import date

from app.observability import configure_logging
from app.voice.limits import VoiceLimits
from app.voice.tickets import TicketStore

ALICE, BOB = uuid.uuid4(), uuid.uuid4()
CONVERSATION = uuid.uuid4()


def test_sessions_per_person() -> None:
    limits = VoiceLimits(max_sessions_per_user=2, audio_seconds_per_day=60)
    limits.opened(ALICE)
    limits.opened(ALICE)
    assert not limits.can_open(ALICE) and limits.can_open(BOB)
    limits.closed(ALICE)
    assert limits.can_open(ALICE) and limits.active(ALICE) == 1


def test_daily_audio_budget_resets_each_day() -> None:
    day = [date(2026, 10, 16)]
    limits = VoiceLimits(max_sessions_per_user=1, audio_seconds_per_day=10, today=lambda: day[0])
    assert limits.charge_audio(ALICE, 6) and limits.charge_audio(ALICE, 4)
    assert not limits.charge_audio(ALICE, 0.1)
    assert limits.charge_audio(BOB, 10)  # per person
    day[0] = date(2026, 10, 17)
    assert limits.charge_audio(ALICE, 10)


def test_unused_tickets_per_person_are_capped() -> None:
    store = TicketStore(ttl_seconds=60, max_per_user=2)
    first = store.issue(ALICE, CONVERSATION)
    second = store.issue(ALICE, CONVERSATION)
    third = store.issue(ALICE, CONVERSATION)  # replaces the oldest
    store.issue(BOB, CONVERSATION)
    assert store.redeem(first) is None
    assert store.redeem(second) is not None and store.redeem(third) is not None
    assert len(store) == 1


def test_provider_socket_logs_stay_quiet_at_debug() -> None:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        configure_logging("DEBUG")
        # websockets logs upgrade headers (API keys) and frames at DEBUG.
        assert logging.getLogger("websockets").getEffectiveLevel() >= logging.WARNING
        assert not logging.getLogger("websockets.client").isEnabledFor(logging.DEBUG)
    finally:
        root.handlers, root.level = handlers, level
        logging.getLogger("websockets").setLevel(logging.NOTSET)

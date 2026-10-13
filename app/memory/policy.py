"""Memory eligibility policy (spec §10.3, §10.6), in application code.

The extraction model proposes candidates; this module decides, per
candidate, to store it, store it and supersede an older memory, hold it for
the person's confirmation, or reject it, and always says why. Rules:

1. Only what the person explicitly said: `explicit` must be true and the
   `evidence` quote must appear in their latest message. An inference is
   never stored as a fact.
2. Identifiers and secrets (phone numbers, addresses, passwords...) are never
   stored. Other sensitive categories are rejected unless the person turned
   on `allow_sensitive` in their memory settings. The category comes from the
   model; as a backstop, content with a precise amount of money is treated
   as financial detail whatever the model said.
3. Content that tries to instruct the assistant ("ignore your rules") is
   rejected; memories are data, never instructions.
4. Duplicates of an active memory are rejected.
5. An update to an existing memory supersedes it; a conflict that does not
   clearly supersede it is held as `pending_confirmation` and is not used
   until the person confirms it.
6. Lifetime sets the expiry (short ~ weeks, medium ~ months, long = the
   retention period); at most `max_per_turn` memories are stored per turn.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from app.memory.schemas import ExtractedCandidate, Memory, MemoryStatus

MAX_CONTENT = 200
LIFETIME_DAYS = {"short": 30, "medium": 180}
_INSTRUCTION_LIKE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}\b(?:instructions?|rules|guidelines|"
    r"system prompt|policy|safety)\b|\bsystem prompt\b|\bjailbreak\b|\bdeveloper mode\b"
    # Demands for agreement are not preferences ("always agree I'm right"),
    # nor is switching off safety behavior ("no safety warnings").
    r"|\b(?:always|have to|must)\s+agree\b|\bnever\b[^.]{0,30}\b(?:disagree|challenge)"
    r"|\b(?:no|stop|without|skip|drop|turn off|remove)\b[^.]{0,20}\b(?:safety|warnings?|"
    r"disclaimers?|filters?)\b",
    re.IGNORECASE,
)

# Backstop for the model's sensitivity label: a precise amount of money (a
# number together with a currency or money word) is financial detail.
_PRECISE_MONEY = re.compile(
    r"(?=.*\d)(?=.*(?:[£$€]|\b(?:salary|salaries|paid|pay|earn(?:s|ing)?|income|debts?|loans?|"
    r"mortgage|overdraft|owe[sd]?|savings|balance|rent|pounds|dollars|euros|grand)\b))",
    re.IGNORECASE,
)

Action = Literal["store", "supersede", "hold", "reject"]


@dataclass(frozen=True)
class MemorySettings:
    memory_enabled: bool = True
    allow_sensitive: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    candidate: ExtractedCandidate
    action: Action
    reason: str
    content: str = ""
    importance: int = 2
    expires_at: datetime | None = None
    # The existing memory this one supersedes (supersede) or conflicts with (hold).
    related_id: uuid.UUID | None = None

    @property
    def status(self) -> MemoryStatus | None:
        if self.action in ("store", "supersede"):
            return MemoryStatus.ACTIVE
        if self.action == "hold":
            return MemoryStatus.PENDING_CONFIRMATION
        return None


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _clean(text: str) -> str:
    text = "".join(c for c in text if c.isprintable())
    return " ".join(text.split())[:MAX_CONTENT]


def decide(
    candidates: Sequence[ExtractedCandidate],
    *,
    latest_message: str,
    shown: Sequence[Memory],
    settings: MemorySettings,
    now: datetime,
    retention_days: int,
    max_per_turn: int = 3,
) -> list[PolicyDecision]:
    """One decision per candidate, in input order."""
    message = normalize(latest_message)
    active = {normalize(m.content): m for m in shown if m.status is MemoryStatus.ACTIVE}
    seen: set[str] = set()
    accepted = 0
    # Most important first, so the per-turn limit drops the least important.
    ranked = sorted(range(len(candidates)), key=lambda i: -candidates[i].importance)
    results: dict[int, PolicyDecision] = {}
    for index in ranked:
        c = candidates[index]

        def reject(reason: str, c: ExtractedCandidate = c) -> PolicyDecision:
            return PolicyDecision(candidate=c, action="reject", reason=reason)

        content = _clean(c.content)
        evidence = normalize(c.evidence)
        related = shown[c.related_memory] if 0 <= c.related_memory < len(shown) else None
        key = normalize(content)
        if not content:
            results[index] = reject("empty")
        elif not settings.memory_enabled:
            results[index] = reject("memory_disabled")
        elif not c.explicit:
            results[index] = reject("not_explicit")
        elif not evidence or evidence not in message:
            results[index] = reject("unsupported_by_message")
        elif c.sensitive_category == "identifier":
            results[index] = reject("identifier")
        elif c.sensitive_category != "none" and not settings.allow_sensitive:
            results[index] = reject(f"sensitive:{c.sensitive_category}")
        elif _PRECISE_MONEY.search(content) and not settings.allow_sensitive:
            results[index] = reject("sensitive:finance")
        elif _INSTRUCTION_LIKE.search(content) or _INSTRUCTION_LIKE.search(c.evidence):
            results[index] = reject("instruction_like")
        elif key in active or key in seen or c.relation == "duplicate":
            results[index] = reject("duplicate")
        elif accepted >= max_per_turn:
            results[index] = reject("turn_limit")
        else:
            days = LIFETIME_DAYS.get(c.lifetime, retention_days)
            expires_at = now + timedelta(days=min(days, retention_days))
            importance = min(3, max(1, c.importance))
            action: Action = "store"
            related_id = None
            if c.relation == "updates" and related is not None:
                action, related_id = "supersede", related.id
            elif c.relation == "conflicts" and related is not None:
                action, related_id = "hold", related.id
            results[index] = PolicyDecision(
                candidate=c,
                action=action,
                reason="eligible" if action == "store" else c.relation,
                content=content,
                importance=importance,
                expires_at=expires_at,
                related_id=related_id,
            )
            accepted += 1
            seen.add(key)
    return [results[index] for index in range(len(candidates))]

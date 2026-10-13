"""Memory eligibility policy (spec §10.3, §10.6), in application code.

The extraction model proposes candidates; this module decides, per
candidate, to store it, store it and supersede an older memory, hold it for
the person's confirmation, or reject it, and always says why. Rules:

1. Only what the person explicitly said: `explicit` must be true, the
   `evidence` quote (at least two words) must appear, on word boundaries, in
   their latest message, and the memory must share a meaningful word with
   the quote. An inference is never stored as a fact.
2. Identifiers and secrets (phone numbers, addresses, emails, passwords...)
   are never stored. Other sensitive categories are rejected unless the
   person turned on `allow_sensitive` in their memory settings. The category
   comes from the model; as a backstop, code also rejects content that
   looks like an identifier or a precise amount of money, whatever the
   model said.
3. Content that tries to instruct the assistant ("ignore your rules", "no
   safety warnings", "always agree with me") is rejected; memories are
   data, never instructions. Text is NFKC-normalized first, and a word that
   mixes alphabets (a Cyrillic letter inside a Latin word) is rejected.
4. Duplicates of an active memory shown to the extractor are rejected.
5. An update to an existing memory supersedes it; a conflict that does not
   clearly supersede it is held as `pending_confirmation`. Pending memories
   reach the reply prompt only marked "unconfirmed", so the companion asks.
6. Lifetime sets the expiry (short ~ weeks, medium ~ months, long = the
   retention period); at most `max_per_turn` memories are stored per turn.

These are backstops for the extraction model's judgement, not a complete
filter: a determined phrasing can still get past a pattern.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from app.memory.schemas import ExtractedCandidate, Memory, MemoryStatus

MAX_CONTENT = 200
LIFETIME_DAYS = {"short": 30, "medium": 180}
_INSTRUCTION_LIKE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|set aside|drop|turn off|disable|get rid of)\b"
    r"[^.]{0,40}\b(?:instructions?|rules|guidelines|system prompt|polic(?:y|ies)|safety|"
    r"restrictions?|limits|filters?|boundaries|everything (?:you were|you've been|you are) told)\b"
    r"|\bignore everything\b|\bsystem prompt\b|\bjailbreak\b|\bdeveloper mode\b"
    r"|\b(?:act as|pretend to be|role-?play as|you are now)\b"
    r"|\b(?:unrestricted|uncensored|unfiltered)\b|\bno (?:restrictions|limits|filters|rules)\b"
    # Demands for agreement are not preferences ("always agree I'm right").
    r"|\b(?:always|have to|must)\s+agree\b"
    r"|\bnever\b[^.]{0,30}\b(?:disagree|challenge|push back|question)"
    # Nor is switching off safety behavior.
    r"|\b(?:no|stop|without|skip|drop|turn off|remove)\b[^.]{0,20}\b(?:safety|warnings?|"
    r"disclaimers?|filters?)\b"
    r"|\b(?:don'?t|do not|doesn'?t|does not|never)\b[^.]{0,30}\b(?:told|tell|mention|suggest|"
    r"recommend|refer)\w*\b[^.]{0,30}\b(?:crisis|hotlines?|helplines?|emergency|samaritans|"
    r"therapy|therapists?|doctors?)",
    re.IGNORECASE,
)

# Backstops for the model's sensitivity label. A number together with a
# currency or money word is financial detail.
_MONEY_WORDS = re.compile(
    r"[£$€₹¥]|\b(?:salary|salaries|paid|pay|earn(?:s|ing)?|makes?|making|income|debts?|loans?|"
    r"mortgage|overdraft|owe[sd]?|savings|balance|rent|pounds|dollars|euros|rupees|grand|"
    r"(?:a|per) (?:year|month|annum|hour))\b",
    re.IGNORECASE,
)
# Identifiers and secrets: emails, links, phone-like or card-like numbers,
# UK postcodes, street addresses, and "my password/PIN is ...".
_IDENTIFIER = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\bhttps?://|\bwww\.\S+"
    r"|(?:\+?\d[\s().-]?){7,}"
    r"|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"
    r"|\b\d{1,5}\s+(?:\w+\s+){0,3}(?:street|st|road|rd|lane|ln|avenue|ave|drive|dr|close|way|"
    r"crescent|court|place|terrace|grove|gardens)\b"
    r"|\b(?:password|passcode|pin|sort code|account number|iban|passport|national insurance|"
    r"social security|ssn)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "having",
        "he",
        "her",
        "his",
        "i",
        "i'm",
        "i've",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
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
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c.isprintable())
    return " ".join(text.split())[:MAX_CONTENT]


def _scripts(word: str) -> set[str]:
    return {
        unicodedata.name(c, "").split(" ")[0] for c in word if c.isalpha() and not c.isascii()
    } | ({"LATIN"} if any(c.isascii() and c.isalpha() for c in word) else set())


def _mixed_script(text: str) -> bool:
    """A word mixing alphabets (a Cyrillic letter inside "ignore") is a disguise."""
    return any(len(_scripts(word)) > 1 for word in text.split())


def looks_financial(text: str) -> bool:
    """A number together with a currency or money word."""
    return any(c.isdigit() for c in text) and _MONEY_WORDS.search(text) is not None


def _words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in _STOPWORDS]


def _supported(content: str, evidence: str, message: str) -> bool:
    """The quote is in the message (whole words, at least two), and the memory
    shares a word with the quote (or a common stem: "live"/"lives")."""
    quote = normalize(evidence)
    if len(quote.split()) < 2 or f" {quote} " not in f" {message} ":
        return False
    quoted = _words(evidence)
    return any(_same_stem(a, b) for a in _words(content) for b in quoted)


def _same_stem(a: str, b: str) -> bool:
    """Equal, a shared 4-letter start ("live"/"lives"), or a short word that
    starts the other ("vet"/"veterinarian")."""
    short, long = sorted((a, b), key=len)
    return (
        a == b
        or (len(short) >= 4 and a[:4] == b[:4])
        or (len(short) >= 3 and long.startswith(short))
    )


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
        related = shown[c.related_memory] if 0 <= c.related_memory < len(shown) else None
        key = normalize(content)
        if not content:
            results[index] = reject("empty")
        elif not settings.memory_enabled:
            results[index] = reject("memory_disabled")
        elif not c.explicit:
            results[index] = reject("not_explicit")
        elif not _supported(content, c.evidence, message):
            results[index] = reject("unsupported_by_message")
        elif c.sensitive_category == "identifier" or _IDENTIFIER.search(content):
            results[index] = reject("identifier")
        elif c.sensitive_category != "none" and not settings.allow_sensitive:
            results[index] = reject(f"sensitive:{c.sensitive_category}")
        elif looks_financial(content) and not settings.allow_sensitive:
            results[index] = reject("sensitive:finance")
        elif _mixed_script(content) or _mixed_script(c.evidence):
            results[index] = reject("mixed_script")
        elif _INSTRUCTION_LIKE.search(content) or _INSTRUCTION_LIKE.search(_clean(c.evidence)):
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

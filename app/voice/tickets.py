"""Single-use tickets for voice WebSocket connections.

Browsers cannot set an Authorization header on a WebSocket, and a bearer
token in a URL ends up in logs. So the client asks for a ticket over the
authenticated HTTP API (POST /v1/conversations/{id}/voice/tickets), then
connects with `?ticket=`. A ticket names one user and one conversation,
works once, and expires after VOICE_TICKET_TTL_SECONDS.

Tickets live in this process's memory: with several workers, the ticket
request and the WebSocket must reach the same one (or this store must move
to shared storage).
"""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    expires_at: float


class TicketStore:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._tickets: dict[str, Ticket] = {}

    def issue(self, user_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
        self._purge()
        token = secrets.token_urlsafe(32)
        self._tickets[token] = Ticket(user_id, conversation_id, time.monotonic() + self._ttl)
        return token

    def redeem(self, token: str) -> Ticket | None:
        """The ticket, removed (single use), or None if unknown or expired."""
        ticket = self._tickets.pop(token, None)
        if ticket is None or ticket.expires_at < time.monotonic():
            return None
        return ticket

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, ticket in self._tickets.items() if ticket.expires_at < now]:
            del self._tickets[token]

    def __len__(self) -> int:
        return len(self._tickets)

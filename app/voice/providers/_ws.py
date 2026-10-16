"""WebSocket helpers shared by the speech providers."""

from __future__ import annotations

import json
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from app.voice.base import SpeechError

OPEN_TIMEOUT_SECONDS = 10.0


async def open_socket(
    provider: str,
    url: str,
    headers: dict[str, str],
    subprotocols: list[str] | None = None,
) -> ClientConnection:
    """Connect, mapping failures to SpeechError. Proxy settings come from the
    environment (websockets' default)."""
    try:
        return await connect(
            url,
            additional_headers=headers,
            subprotocols=subprotocols,  # type: ignore[arg-type]
            open_timeout=OPEN_TIMEOUT_SECONDS,
            max_size=8 * 1024 * 1024,
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        raise SpeechError(
            provider, f"http_{status}", retryable=status in (408, 429) or status >= 500
        ) from None
    except (InvalidHandshake, OSError, TimeoutError) as exc:
        raise SpeechError(provider, type(exc).__name__, retryable=True) from None


def closed_error(provider: str, exc: ConnectionClosed) -> SpeechError:
    code = exc.rcvd.code if exc.rcvd is not None else None
    reason = exc.rcvd.reason if exc.rcvd is not None else ""
    return SpeechError(provider, f"closed_{code}{':' + reason if reason else ''}", retryable=True)


async def send_json(socket: ClientConnection, payload: dict[str, Any]) -> None:
    await socket.send(json.dumps(payload))


def parse(message: str | bytes) -> dict[str, Any] | None:
    if isinstance(message, bytes):
        return None
    try:
        value = json.loads(message)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

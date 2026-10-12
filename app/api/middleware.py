"""Pure-ASGI middleware: request tracing and request-body size limits."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability import log_event, trace_id_var

logger = logging.getLogger("app.http")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def json_error(status: int, category: str, code: str, message: str) -> list[Message]:
    body = json.dumps(
        {
            "error": {
                "category": category,
                "code": code,
                "message": message,
                "trace_id": trace_id_var.get(),
            }
        }
    ).encode()
    return [
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        },
        {"type": "http.response.body", "body": body},
    ]


def _innermost_location(exc: BaseException) -> str | None:
    tb = exc.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    return f"{tb.tb_frame.f_code.co_filename.rsplit('/', 1)[-1]}:{tb.tb_lineno}"


class TraceMiddleware:
    """Assigns a trace ID (honoring a well-formed inbound X-Request-ID), logs
    request completion without bodies, and converts unhandled exceptions into
    a generic 500 that carries the trace ID (type and location are logged;
    messages and tracebacks are not, since they can embed request data)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = None
        for name, value in scope["headers"]:
            if name == b"x-request-id":
                inbound = value.decode("latin-1")
                break
        trace_id = inbound if inbound and _VALID_REQUEST_ID.match(inbound) else uuid.uuid4().hex
        token = trace_id_var.set(trace_id)
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_trace(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)["x-request-id"] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        except Exception as exc:
            log_event(
                logger,
                "request.unhandled_error",
                level=logging.ERROR,
                exc_type=type(exc).__name__,
                location=_innermost_location(exc),
            )
            if response_started:
                raise
            for message in json_error(
                500, "internal_error", "internal_error", "An internal error occurred."
            ):
                await send_with_trace(message)
        finally:
            log_event(
                logger,
                "request.completed",
                method=scope["method"],
                path=scope["path"],
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            trace_id_var.reset(token)


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects request bodies above `max_bytes` with 413, checking both the
    declared Content-Length and the bytes actually streamed."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = self.max_bytes + 1
                if declared > self.max_bytes:
                    await self._reject(send)
                    return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # Frameworks may convert the read error into their own response
            # (FastAPI answers 400); replace it with a 413.
            nonlocal response_started
            if exceeded:
                if message["type"] == "http.response.start" and not response_started:
                    response_started = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        for message in json_error(
            413,
            "invalid_request",
            "request_too_large",
            f"Request body exceeds {self.max_bytes} bytes.",
        ):
            await send(message)

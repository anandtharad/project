"""Structured, privacy-conscious logging.

Events are emitted as single-line JSON with a trace ID. Callers pass explicit
fields; message and memory *content* must never be passed as a field.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_FIELDS_ATTR = "event_fields"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        trace_id = trace_id_var.get()
        if trace_id is not None:
            payload["trace_id"] = trace_id
        fields = getattr(record, _FIELDS_ATTR, None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            # Exception type only: tracebacks can embed request data.
            exc_type = record.exc_info[0]
            payload["exc_type"] = exc_type.__name__ if exc_type else None
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # SDK/HTTP client debug logs include request bodies (system prompt and
    # user messages), so they are capped at WARNING even when LOG_LEVEL=DEBUG.
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "anthropic",
        "openai",
        "groq",
        "google_genai",
        "google.genai",
        "typesafe_sdk",
    ):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    # The trace ID is taken when the event happens, not when it is formatted:
    # a handler may format later, outside the request's context.
    trace_id = trace_id_var.get()
    if trace_id is not None and "trace_id" not in fields:
        fields = {"trace_id": trace_id, **fields}
    logger.log(level, event, extra={_FIELDS_ATTR: fields})

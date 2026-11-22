"""LangChain's tracing client (langsmith) comes with langchain-core, which
LangGraph requires. It sends traces, conversation content included, only
when tracing is switched on by environment variables. This app never turns
it on; switching it on in production is refused, elsewhere warned about."""

from __future__ import annotations

import logging
import os

from app.config import AppEnv, Settings
from app.observability import log_event

logger = logging.getLogger("app.agent")

TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def tracing_enabled() -> list[str]:
    return [
        name
        for name in TRACING_VARIABLES
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
    ]


def warn_if_tracing(settings: Settings) -> None:
    enabled = tracing_enabled()
    if not enabled:
        return
    if settings.app_env is AppEnv.PRODUCTION:
        raise ValueError(
            f"{', '.join(enabled)} would send conversation content to LangSmith; "
            "turn it off in production"
        )
    log_event(logger, "agent.tracing_enabled", level=logging.WARNING, variables=enabled)

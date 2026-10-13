"""Post-turn background work (spec §9.2 step 14).

Tasks run after the reply has been persisted and returned. A failure is
logged by type only and never reaches the completed turn. On shutdown,
pending tasks get a short grace period and are then cancelled.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from app.observability import log_event

logger = logging.getLogger("app.background")


class BackgroundTasks:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        task = asyncio.create_task(self._guard(coro, name), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guard(self, coro: Coroutine[Any, Any, Any], name: str) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Type only: messages can embed conversation content.
            log_event(
                logger,
                "background.failed",
                level=logging.ERROR,
                task=name,
                error_type=type(exc).__name__,
            )

    async def drain(self) -> None:
        """Wait for every pending task (tests, graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self, grace_seconds: float = 5.0) -> None:
        if not self._tasks:
            return
        _, pending = await asyncio.wait(list(self._tasks), timeout=grace_seconds)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

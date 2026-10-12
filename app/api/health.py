from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. Does not touch dependencies."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: database reachable and schema migrated to the expected head."""
    checks: dict[str, Any] = {}
    expected_head: str = request.app.state.migration_head
    try:
        async with asyncio.timeout(3):
            async with request.app.state.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                checks["database"] = "ok"
                current = (
                    await conn.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one_or_none()
        checks["migrations"] = "ok" if current == expected_head else "pending"
    except (SQLAlchemyError, OSError):
        checks.setdefault("database", "unavailable")
        checks["migrations"] = "unknown"
    checks["llm_provider"] = request.app.state.llm.provider
    checks["prompt"] = request.app.state.prompt.key
    is_ready = checks["database"] == "ok" and checks["migrations"] == "ok"
    return JSONResponse(
        {"status": "ready" if is_ready else "not_ready", "checks": checks},
        status_code=200 if is_ready else 503,
    )

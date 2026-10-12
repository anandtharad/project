"""HTTP error contract.

Every error response has the shape:
    {"error": {"category": ..., "code": ..., "message": ..., "trace_id": ...}}
Stack traces, secrets, and submitted input are never included. Unhandled
exceptions are converted to a generic 500 by `TraceMiddleware`.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import AppError, ErrorCategory
from app.observability import log_event, trace_id_var

logger = logging.getLogger("app.errors")

_HTTP_CATEGORY = {
    400: ErrorCategory.INVALID_REQUEST,
    401: ErrorCategory.UNAUTHORIZED,
    403: ErrorCategory.FORBIDDEN,
    404: ErrorCategory.NOT_FOUND,
    405: ErrorCategory.INVALID_REQUEST,
    409: ErrorCategory.CONFLICT,
    413: ErrorCategory.INVALID_REQUEST,
    415: ErrorCategory.INVALID_REQUEST,
    429: ErrorCategory.RATE_LIMITED,
}


def error_response(
    *,
    status: int,
    category: ErrorCategory,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "category": category.value,
        "code": code,
        "message": message,
        "trace_id": trace_id_var.get(),
    }
    if details:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


async def _app_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    headers = (
        {"Retry-After": str(exc.retry_after_seconds)}
        if exc.retry_after_seconds is not None
        else None
    )
    if exc.category is ErrorCategory.UNAUTHORIZED:
        headers = {**(headers or {}), "WWW-Authenticate": "Bearer"}
    return error_response(
        status=exc.http_status,
        category=exc.category,
        code=exc.code,
        message=exc.message,
        headers=headers,
    )


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only field locations and messages: Pydantic's "input"/"ctx" may echo content.
    details = [
        {"loc": [str(part) for part in err.get("loc", ())], "msg": str(err.get("msg", ""))}
        for err in exc.errors()
    ]
    log_event(logger, "request.validation_failed", fields=[d["loc"] for d in details])
    return error_response(
        status=422,
        category=ErrorCategory.INVALID_REQUEST,
        code="validation_failed",
        message="The request is invalid.",
        details=details,
    )


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    category = _HTTP_CATEGORY.get(exc.status_code, ErrorCategory.INTERNAL_ERROR)
    code = {404: "route_not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) and exc.status_code < 500 else "Error."
    return error_response(status=exc.status_code, category=category, code=code, message=message)


async def _dependency_error(_: Request, exc: Exception) -> JSONResponse:
    # SQLAlchemy errors and raw connection errors (OSError) from the driver.
    code = "database_unavailable" if isinstance(exc, SQLAlchemyError) else "dependency_unavailable"
    log_event(logger, "dependency.unavailable", level=logging.ERROR, exc_type=type(exc).__name__)
    return error_response(
        status=503,
        category=ErrorCategory.DEPENDENCY_UNAVAILABLE,
        code=code,
        message="A required service is unavailable. Please retry.",
        headers={"Retry-After": "2"},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(SQLAlchemyError, _dependency_error)
    app.add_exception_handler(OSError, _dependency_error)

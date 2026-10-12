"""Application error contract shared by the API and the orchestration layer."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


DEFAULT_HTTP_STATUS: dict[ErrorCategory, int] = {
    ErrorCategory.INVALID_REQUEST: 400,
    ErrorCategory.UNAUTHORIZED: 401,
    ErrorCategory.FORBIDDEN: 403,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.RATE_LIMITED: 429,
    ErrorCategory.DEPENDENCY_UNAVAILABLE: 503,
    ErrorCategory.TIMEOUT: 504,
    ErrorCategory.INTERNAL_ERROR: 500,
}


class AppError(Exception):
    """An error safe to return to clients.

    `message` must never contain secrets, stack traces, or another user's data.
    """

    def __init__(
        self,
        category: ErrorCategory,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message
        self.http_status = http_status or DEFAULT_HTTP_STATUS[category]
        self.retry_after_seconds = retry_after_seconds


def conversation_not_found() -> AppError:
    # Used both for missing conversations and for conversations owned by
    # someone else, so existence is never disclosed across users.
    return AppError(ErrorCategory.NOT_FOUND, "conversation_not_found", "Conversation not found.")

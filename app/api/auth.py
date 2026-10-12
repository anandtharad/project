"""DEVELOPMENT AUTHENTICATION - NOT FOR PRODUCTION.

Bearer tokens are matched against the static DEV_AUTH_TOKENS mapping
("handle:token" pairs) and mapped to a user row keyed by `dev:<handle>`.
There is no token issuance, expiry, rotation, or revocation. Configuration
refuses to start with APP_ENV=production while this is the only auth mode.
"""

from __future__ import annotations

import hmac
import uuid
from dataclasses import dataclass

from fastapi import Request

from app.db import repositories as repo
from app.errors import AppError, ErrorCategory


@dataclass(frozen=True)
class AuthenticatedUser:
    id: uuid.UUID
    external_id: str


def _unauthorized() -> AppError:
    return AppError(ErrorCategory.UNAUTHORIZED, "unauthorized", "Valid credentials are required.")


def resolve_dev_token(token: str, token_map: dict[str, str]) -> str | None:
    """Constant-time comparison against every configured token."""
    match: str | None = None
    for candidate, handle in token_map.items():
        if hmac.compare_digest(candidate.encode(), token.encode()):
            match = handle
    return match


async def authenticate(request: Request) -> AuthenticatedUser:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized()
    handle = resolve_dev_token(token.strip(), request.app.state.dev_tokens)
    if handle is None:
        raise _unauthorized()
    external_id = f"dev:{handle}"
    async with request.app.state.session_factory() as session, session.begin():
        user_id = await repo.get_or_create_user(session, external_id)
    return AuthenticatedUser(id=user_id, external_id=external_id)

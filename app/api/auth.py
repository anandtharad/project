"""Authentication (docs/api.md, "Authentication").

Two modes, chosen by AUTH_MODE:

- "clerk" (production): the client signs people in with Clerk and sends the
  Clerk session token as `Authorization: Bearer <token>`. The token is an
  RS256 JWT, verified here against the Clerk instance's public key (a PEM key
  from CLERK_JWT_KEY, or the instance's JWKS from CLERK_JWKS_URL): signature,
  `exp` and `nbf` (with a few seconds of clock skew), `sub`, the authorized
  party `azp` when one is configured, and `iss` when CLERK_ISSUER is set
  (Clerk's manual JWT verification guide). The person is the user row keyed
  by `clerk:<sub>`. Sign-up, login, sessions, expiry and revocation are
  Clerk's; this service never sees a password.
- "dev" (DEVELOPMENT ONLY): bearer tokens are matched against the static
  DEV_AUTH_TOKENS pairs and mapped to the user row keyed by `dev:<handle>`.
  No issuance, expiry, rotation or revocation. Configuration refuses
  APP_ENV=production in this mode.
"""

from __future__ import annotations

import asyncio
import hmac
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Request
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWTError

from app.config import Settings
from app.db import repositories as repo
from app.errors import AppError, ErrorCategory

# Clerk signs session tokens with RS256; nothing else is accepted (no "none",
# no HMAC with the public key as secret).
ALGORITHMS = ["RS256"]
CLOCK_SKEW_SECONDS = 5
JWKS_CACHE_SECONDS = 3600


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


class ClerkVerifier:
    """Verifies Clerk session tokens; returns the user id (`sub`)."""

    def __init__(
        self,
        *,
        jwt_key: str | None,
        jwks_url: str | None,
        authorized_parties: Sequence[str],
        issuer: str | None = None,
    ) -> None:
        if not jwt_key and not jwks_url:
            raise ValueError("a PEM key or a JWKS URL is required")
        self._key = jwt_key.replace("\\n", "\n").strip() if jwt_key else None
        self._jwks = (
            None
            if self._key
            else PyJWKClient(str(jwks_url), cache_keys=True, lifespan=JWKS_CACHE_SECONDS)
        )
        self._parties = {p.rstrip("/") for p in authorized_parties}
        self._issuer = issuer

    @classmethod
    def from_settings(cls, settings: Settings) -> ClerkVerifier:
        return cls(
            jwt_key=settings.clerk_jwt_key,
            jwks_url=settings.clerk_jwks_url,
            authorized_parties=settings.clerk_party_list(),
            issuer=settings.clerk_issuer,
        )

    async def verify(self, token: str) -> str:
        try:
            if self._key is not None:
                key: Any = self._key
            else:
                assert self._jwks is not None
                # Blocking HTTP on a cache miss: keep it off the event loop.
                signing = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
                key = signing.key
            claims = jwt.decode(
                token,
                key,
                algorithms=ALGORITHMS,
                options={"require": ["exp", "nbf", "sub"], "verify_aud": False},
                issuer=self._issuer,
                leeway=CLOCK_SKEW_SECONDS,
            )
        except PyJWKClientConnectionError:
            raise AppError(
                ErrorCategory.DEPENDENCY_UNAVAILABLE,
                "auth_unavailable",
                "Sign-in could not be checked right now; try again shortly.",
                http_status=503,
            ) from None
        except PyJWTError:
            raise _unauthorized() from None
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise _unauthorized()
        # Clerk: compare `azp` with the origins allowed to issue tokens; a
        # token without `azp` passes this step.
        party = claims.get("azp")
        if self._parties and party is not None and str(party).rstrip("/") not in self._parties:
            raise _unauthorized()
        return subject


async def authenticate(request: Request) -> AuthenticatedUser:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized()
    clerk: ClerkVerifier | None = getattr(request.app.state, "clerk", None)
    if clerk is not None:
        external_id = f"clerk:{await clerk.verify(token.strip())}"
    else:
        handle = resolve_dev_token(token.strip(), request.app.state.dev_tokens)
        if handle is None:
            raise _unauthorized()
        external_id = f"dev:{handle}"
    async with request.app.state.session_factory() as session, session.begin():
        user_id = await repo.get_or_create_user(session, external_id)
    return AuthenticatedUser(id=user_id, external_id=external_id)

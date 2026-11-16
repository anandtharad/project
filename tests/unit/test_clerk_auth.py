"""Clerk session-token verification (app/api/auth.py) with a locally
generated RSA key: a PEM key and a JWKS endpoint, expiry, not-before,
authorized party, issuer, algorithm confusion and tampering."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from app.api.auth import ClerkVerifier
from app.config import Settings
from app.errors import AppError

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_PEM = (
    KEY.public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)
ORIGIN = "https://app.example.com"
ISSUER = "https://clerk.example.com"


def token(key: Any = KEY, kid: str = "k1", alg: str = "RS256", **claims: Any) -> str:
    now = int(time.time())
    payload = {"sub": "user_123", "exp": now + 60, "nbf": now - 1, "iat": now, "azp": ORIGIN}
    payload.update(claims)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, key, algorithm=alg, headers={"kid": kid})


def verifier(**overrides: Any) -> ClerkVerifier:
    options: dict[str, Any] = {
        "jwt_key": PUBLIC_PEM,
        "jwks_url": None,
        "authorized_parties": [ORIGIN],
    }
    options.update(overrides)
    return ClerkVerifier(**options)


async def rejected(v: ClerkVerifier, value: str) -> str:
    with pytest.raises(AppError) as error:
        await v.verify(value)
    return error.value.code


async def test_a_valid_token_gives_the_clerk_user_id() -> None:
    assert await verifier().verify(token()) == "user_123"
    # Clerk: a token without `azp` passes the authorized-party step.
    assert await verifier().verify(token(azp=None)) == "user_123"
    # The key as an environment variable with escaped newlines.
    escaped = PUBLIC_PEM.replace("\n", "\\n")
    assert await verifier(jwt_key=escaped).verify(token()) == "user_123"


async def test_invalid_tokens_are_unauthorized() -> None:
    v = verifier()
    now = int(time.time())
    bad = {
        "expired": token(exp=now - 60),
        "not yet valid": token(nbf=now + 60),
        "no subject": token(sub=None),
        "no expiry": token(exp=None),
        "another party": token(azp="https://evil.example.com"),
        "another key": token(key=OTHER_KEY),
        "tampered": token()[:-4] + "AAAA",
        "garbage": "not-a-jwt",
    }
    for name, value in bad.items():
        assert await rejected(v, value) == "unauthorized", name

    # Algorithm confusion: HS256 "signed" with the public key as the secret,
    # built by hand (PyJWT refuses to create it).
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"sub": "attacker", "exp": now + 60, "nbf": now - 1}).encode())
    mac = hmac.new(PUBLIC_PEM.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    assert await rejected(v, f"{head}.{body}.{b64(mac)}") == "unauthorized"
    unsigned = jwt.encode({"sub": "x", "exp": now + 60, "nbf": now - 1}, "", algorithm="none")
    assert await rejected(v, unsigned) == "unauthorized"


async def test_issuer_is_checked_when_configured() -> None:
    v = verifier(issuer=ISSUER)
    assert await v.verify(token(iss=ISSUER)) == "user_123"
    assert await rejected(v, token(iss="https://other.example.com")) == "unauthorized"
    assert await rejected(v, token()) == "unauthorized"  # no iss at all


@pytest.fixture
def jwks_url() -> Iterator[str]:
    jwk = json.loads(RSAAlgorithm.to_jwk(KEY.public_key()))
    body = json.dumps({"keys": [{**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/.well-known/jwks.json"
    finally:
        server.shutdown()


async def test_jwks_keys_are_fetched_and_matched_by_kid(jwks_url: str) -> None:
    v = verifier(jwt_key=None, jwks_url=jwks_url)
    assert await v.verify(token()) == "user_123"
    assert await rejected(v, token(kid="unknown")) == "unauthorized"


async def test_an_unreachable_jwks_is_a_503_not_a_401() -> None:
    v = verifier(jwt_key=None, jwks_url="http://127.0.0.1:9/.well-known/jwks.json")
    assert await rejected(v, token()) == "auth_unavailable"


def test_settings() -> None:
    with pytest.raises(ValueError, match="CLERK_JWT_KEY or CLERK_JWKS_URL"):
        Settings(_env_file=None, auth_mode="clerk")
    with pytest.raises(ValueError, match="AUTH_MODE=clerk"):
        Settings(_env_file=None, app_env="production")
    with pytest.raises(ValueError, match="CLERK_AUTHORIZED_PARTIES"):
        Settings(_env_file=None, app_env="production", auth_mode="clerk", clerk_jwt_key=PUBLIC_PEM)
    settings = Settings(
        _env_file=None,
        auth_mode="clerk",
        clerk_jwt_key=PUBLIC_PEM,
        clerk_authorized_parties=f"{ORIGIN}/, http://localhost:3000",
    )
    assert settings.clerk_party_list() == [ORIGIN, "http://localhost:3000"]

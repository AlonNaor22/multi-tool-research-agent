"""Bearer-token authentication for the REST API."""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ─── Module overview ───────────────────────────────────────────────
# Bearer-token auth as a FastAPI dependency. When API_AUTH_TOKEN is
# unset (dev mode), the dependency is a no-op and an unauthenticated
# request proceeds; when set, every request must carry a matching
# `Authorization: Bearer <token>` header or it's rejected with 401.
# The token is read per-request so tests can toggle the env var via
# monkeypatch without rebuilding the app.
# ───────────────────────────────────────────────────────────────────


# auto_error=False lets us emit a custom 401 message when the header
# is missing instead of FastAPI's default 403/Forbidden surprise.
_bearer = HTTPBearer(auto_error=False, description="API bearer token (set API_AUTH_TOKEN to require)")


# Returns the configured API token, or None if auth is disabled.
def _expected_token() -> Optional[str]:
    """Read the expected token from the env at call time."""
    token = os.getenv("API_AUTH_TOKEN")
    return token if token else None


# FastAPI dependency: validates the bearer token, or short-circuits
# when auth is disabled. Raises 401 on missing / invalid credentials.
async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """Validate Authorization: Bearer <token>; no-op when auth disabled."""
    expected = _expected_token()
    if not expected:
        return None  # dev mode — auth disabled
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Send `Authorization: Bearer <token>`.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def auth_is_enabled() -> bool:
    """Return True iff API_AUTH_TOKEN is set; used by the startup log."""
    return _expected_token() is not None

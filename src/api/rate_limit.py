"""Per-endpoint rate limiting backed by slowapi."""

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ─── Module overview ───────────────────────────────────────────────
# A single slowapi.Limiter instance shared across all routes. The
# default key extracts the remote IP (X-Forwarded-For aware) so each
# caller gets their own bucket. Per-endpoint limits are set via the
# @limiter.limit("N/period") decorator on individual route handlers.
# Tests flip `limiter.enabled = False` to disable enforcement.
#
# rate_limit_handler is a small wrapper around slowapi's default that
# adds a `Retry-After` header so HTTP clients know how long to wait
# before retrying — slowapi's stock handler only emits X-RateLimit-*.
# ───────────────────────────────────────────────────────────────────

# Default rate limits per protected endpoint. Cheap reads get a high
# ceiling; expensive LLM calls are throttled aggressively to keep
# accidental loops from draining the token budget.
LIMIT_QUERY = "10/minute"
LIMIT_QUERY_STREAM = "10/minute"
LIMIT_SESSIONS_READ = "60/minute"
LIMIT_SESSIONS_DELETE = "30/minute"

# Shared limiter. The Limiter remembers per-key counts in-process; for
# multi-worker deployments swap `storage_uri` for a Redis URL.
limiter = Limiter(key_func=get_remote_address, default_limits=[])


# Custom 429 handler: same body as slowapi's default plus a Retry-After
# header derived from the view's rate-limit reset hint when available.
def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return slowapi's standard 429 payload plus a Retry-After header."""
    response = JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"},
        status_code=429,
    )
    # Inject the X-RateLimit-* headers that slowapi already supports.
    view_limit = getattr(request.state, "view_rate_limit", None)
    if view_limit is not None:
        response = request.app.state.limiter._inject_headers(response, view_limit)
    # Fallback Retry-After: clients need it to back off cleanly. Use
    # the period seconds from the limit string ("10/minute" -> 60).
    if "retry-after" not in {k.lower() for k in response.headers}:
        retry_seconds = _retry_after_seconds(exc.detail)
        response.headers["Retry-After"] = str(retry_seconds)
    return response


# Maps slowapi's "N per P unit" detail string to a Retry-After in seconds.
def _retry_after_seconds(detail: str) -> int:
    """Best-effort: parse '10 per 1 minute' → 60. Falls back to 60s."""
    parts = detail.split()
    # Expected form: "<N> per <P> <unit>" — e.g. "10 per 1 minute"
    try:
        unit = parts[-1].rstrip("s").lower()
        scale = parts[-2]
        per_seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        return int(scale) * per_seconds
    except (ValueError, KeyError, IndexError):
        return 60

"""FastAPI dependency providers and shared helpers."""

import asyncio
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

from src.constants import (
    MODE_AUTO, MODE_DIRECT, MODE_PLAN_EXECUTE, MODE_MULTI_AGENT,
)

if TYPE_CHECKING:
    from src.agent import ResearchAgent

# ─── Module overview ───────────────────────────────────────────────
# Dependency providers used by route handlers. get_agent() pulls the
# singleton agent off app.state (populated by the lifespan handler).
# agent_lock() returns the per-app asyncio.Lock that serializes any
# operation mutating agent state (current_session_id, callbacks).
# ───────────────────────────────────────────────────────────────────

API_MODE_TO_INTERNAL = {
    "auto": MODE_AUTO,
    "direct": MODE_DIRECT,
    "plan": MODE_PLAN_EXECUTE,
    "multi": MODE_MULTI_AGENT,
}


def get_agent(request: Request) -> "ResearchAgent":
    """Return the singleton ResearchAgent from app state."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized.",
        )
    return agent


def get_agent_lock(request: Request) -> asyncio.Lock:
    """Return the per-app asyncio.Lock that serializes agent state mutations."""
    lock = getattr(request.app.state, "agent_lock", None)
    if lock is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent lock not initialized.",
        )
    return lock

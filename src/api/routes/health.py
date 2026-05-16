"""Health and tool-status endpoints."""

from fastapi import APIRouter, Depends

from src.agent import ResearchAgent
from src.api.dependencies import get_agent
from src.api.models import HealthResponse

# ─── Module overview ───────────────────────────────────────────────
# Read-only endpoints for service status: GET /health reports overall
# state plus the per-tool enabled/disabled split derived from the
# agent's tool-health probe.
# ───────────────────────────────────────────────────────────────────

router = APIRouter(tags=["health"])


# Returns service health and the enabled/disabled tool split.
# Status is "ok" when no tools are disabled, "degraded" otherwise.
@router.get("/health", response_model=HealthResponse)
def health(agent: ResearchAgent = Depends(get_agent)) -> HealthResponse:
    """Return service health plus the agent's enabled/disabled tool split."""
    enabled = [t.name for t in agent.tools]
    disabled = list(agent.disabled_tools)
    return HealthResponse(
        status="degraded" if disabled else "ok",
        tool_count=len(enabled) + len(disabled),
        enabled_tools=enabled,
        disabled_tools=disabled,
    )

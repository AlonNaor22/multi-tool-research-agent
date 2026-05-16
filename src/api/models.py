"""Pydantic request/response models for the REST API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ─── Module overview ───────────────────────────────────────────────
# Pydantic models for every API request and response. The Literal
# `mode` field constrains the agent execution mode at the schema
# boundary so the route handlers never see unknown values.
# ───────────────────────────────────────────────────────────────────

# API mode names are short/lowercase; route_query maps these to the
# internal MODE_* constants.
ModeName = Literal["auto", "direct", "plan", "multi"]


class QueryRequest(BaseModel):
    """Body of POST /query and /query/stream."""
    query: str = Field(min_length=1, description="Research question to ask the agent.")
    mode: ModeName = Field(
        default="auto",
        description=(
            "Execution mode. 'auto' picks direct vs plan based on the question; "
            "'direct' is a single-agent run; 'plan' uses plan-and-execute; "
            "'multi' uses the multi-agent orchestrator."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional caller-supplied session id. If omitted, the agent generates "
            "a new one; the response always returns the session_id used."
        ),
    )


class QueryResponse(BaseModel):
    """Body of POST /query (non-streaming)."""
    answer: str
    session_id: str
    tokens_used: int
    duration_seconds: float


class Exchange(BaseModel):
    """One question/answer pair from a session's history."""
    question: str
    answer: str


class SessionSummary(BaseModel):
    """One entry in GET /sessions."""
    session_id: str
    created_at: str
    message_count: int


class SessionDetail(BaseModel):
    """Body of GET /sessions/{session_id}."""
    session_id: str
    exchanges: List[Exchange]


class DeleteResult(BaseModel):
    """Body of DELETE /sessions/{session_id}."""
    session_id: str
    deleted: bool


class HealthResponse(BaseModel):
    """Body of GET /health."""
    status: Literal["ok", "degraded"]
    tool_count: int
    enabled_tools: List[str]
    disabled_tools: List[str]


class ErrorResponse(BaseModel):
    """Standard error envelope for non-2xx responses."""
    error: str
    detail: Optional[str] = None

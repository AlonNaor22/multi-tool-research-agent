"""Session listing, retrieval, and deletion endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.agent import ResearchAgent
from src.api.auth import verify_token
from src.api.dependencies import get_agent
from src.api.models import DeleteResult, Exchange, SessionDetail, SessionSummary
from src.api.rate_limit import LIMIT_SESSIONS_DELETE, LIMIT_SESSIONS_READ, limiter
from src.session_manager import delete_session, list_sessions, load_session

# ─── Module overview ───────────────────────────────────────────────
# Session endpoints backed by the AsyncSqliteSaver checkpoint DB.
# List/load wrap src.session_manager helpers; delete shells out to
# the same helper. Async I/O on the saver is offloaded via FastAPI's
# default threadpool because the helpers are sync-only. Bearer-token
# auth + slowapi rate limits gate every request here.
# ───────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
    dependencies=[Depends(verify_token)],
)


# Returns every saved session as a summary (id, created_at, message count).
@router.get("", response_model=List[SessionSummary])
@limiter.limit(LIMIT_SESSIONS_READ)
def list_all_sessions(
    request: Request,  # noqa: ARG001 — slowapi reads `request` for IP-based keying
    agent: ResearchAgent = Depends(get_agent),
) -> List[SessionSummary]:
    """List every session stored in the checkpoint database."""
    rows = list_sessions(agent.checkpointer)
    return [SessionSummary(**row) for row in rows]


# Returns full Q/A history for a session; 404 if no checkpoint exists.
@router.get("/{session_id}", response_model=SessionDetail)
@limiter.limit(LIMIT_SESSIONS_READ)
def get_session(
    request: Request,  # noqa: ARG001 — slowapi reads `request` for IP-based keying
    session_id: str,
    agent: ResearchAgent = Depends(get_agent),
) -> SessionDetail:
    """Return the conversation history for a specific session."""
    history = load_session(agent.checkpointer, session_id)
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return SessionDetail(
        session_id=session_id,
        exchanges=[Exchange(question=q, answer=a) for q, a in history],
    )


# Deletes a session's checkpoints. 404 if it didn't exist.
@router.delete("/{session_id}", response_model=DeleteResult)
@limiter.limit(LIMIT_SESSIONS_DELETE)
def remove_session(
    request: Request,  # noqa: ARG001 — slowapi reads `request` for IP-based keying
    session_id: str,
    agent: ResearchAgent = Depends(get_agent),
) -> DeleteResult:
    """Delete every checkpoint for a session thread."""
    deleted = delete_session(agent.checkpointer, session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return DeleteResult(session_id=session_id, deleted=True)

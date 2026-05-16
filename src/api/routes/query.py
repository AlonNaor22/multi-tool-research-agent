"""Query endpoints — POST /query (blocking) and POST /query/stream (SSE)."""

import asyncio
import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from src.agent import ResearchAgent
from src.api.dependencies import API_MODE_TO_INTERNAL, get_agent, get_agent_lock
from src.api.models import QueryRequest, QueryResponse
from src.constants import (
    EVENT_DONE, MODE_AUTO, MODE_DIRECT, MODE_MULTI_AGENT, MODE_PLAN_EXECUTE,
)
from src.planner import is_simple_query
from src.rate_limiter import RateLimitExceeded
from src.session_manager import generate_session_id

# ─── Module overview ───────────────────────────────────────────────
# Two endpoints that drive the agent: /query returns the final answer
# after the run completes; /query/stream emits Server-Sent Events.
# Concurrent requests are serialized by app.state.agent_lock because
# the ResearchAgent owns mutable state (current_session_id, callbacks,
# rate limiter) that would race under parallel access.
# ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# Sets the agent's session_id to the caller's value, or a freshly generated
# one if absent. Returns the session_id that ended up active.
def _scope_session(agent: ResearchAgent, requested_id: str | None) -> str:
    """Scope the agent to a session id (caller-supplied or freshly generated)."""
    session_id = requested_id or generate_session_id()
    agent.set_session_id(session_id)
    return session_id


# Maps API mode → internal MODE_* constant and dispatches the right
# async agent method. Returns the final answer string.
async def _run_query_for_mode(agent: ResearchAgent, query: str, mode: str) -> str:
    """Dispatch to the right async agent method for the requested mode."""
    internal_mode = API_MODE_TO_INTERNAL[mode]

    if internal_mode == MODE_MULTI_AGENT:
        return await agent.multi_agent_query(query, verbose=False)
    if internal_mode == MODE_PLAN_EXECUTE:
        return await agent.plan_and_execute(query, verbose=False)
    if internal_mode == MODE_AUTO and not is_simple_query(query):
        return await agent.plan_and_execute(query, verbose=False)
    # Auto-simple or explicit direct
    return await agent.query(query, show_timing=False)


@router.post("/query", response_model=QueryResponse)
async def run_query(
    body: QueryRequest,
    agent: ResearchAgent = Depends(get_agent),
    lock: asyncio.Lock = Depends(get_agent_lock),
) -> QueryResponse:
    """Run a query to completion and return the final answer plus metrics."""
    async with lock:
        session_id = _scope_session(agent, body.session_id)
        start = time.monotonic()
        try:
            answer = await _run_query_for_mode(agent, body.query, body.mode)
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e),
            )
        except Exception as e:
            logger.exception("Query failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {e}",
            )

        duration = time.monotonic() - start
        metrics = agent.get_last_metrics()
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            tokens_used=getattr(metrics, "total_tokens", 0) or 0,
            duration_seconds=round(duration, 3),
        )


# Bridges a sync generator (route_query yields events on the persistent IO
# loop) to async iteration by pulling each next() in the default executor.
async def _bridge_sync_iter(sync_gen) -> AsyncIterator:
    """Pull items from a sync generator on the default executor."""
    loop = asyncio.get_event_loop()
    sentinel = object()
    while True:
        item = await loop.run_in_executor(None, next, sync_gen, sentinel)
        if item is sentinel:
            return
        yield item


@router.post("/query/stream")
async def stream_query(
    body: QueryRequest,
    agent: ResearchAgent = Depends(get_agent),
    lock: asyncio.Lock = Depends(get_agent_lock),
):
    """Run a query and stream typed events as Server-Sent Events.

    Each SSE message has `event` set to the agent event type (e.g.
    `synthesis_token`, `step_started`, `done`) and `data` set to the
    JSON-encoded event payload. The final `done` event carries the
    full answer and the session_id used for the run.
    """
    # Acquire the lock OUTSIDE the generator so concurrent stream
    # requests queue cleanly. The generator releases the lock on its
    # finally branch (when SSE client disconnects or stream completes).
    await lock.acquire()
    session_id = _scope_session(agent, body.session_id)

    async def event_stream():
        try:
            internal_mode = API_MODE_TO_INTERNAL[body.mode]
            sync_gen = agent.route_query(body.query, mode=internal_mode)

            saw_done = False
            try:
                async for event in _bridge_sync_iter(sync_gen):
                    payload = {**event, "session_id": session_id}
                    if event.get("type") == EVENT_DONE:
                        saw_done = True
                    yield {"event": event.get("type", "message"), "data": json.dumps(payload)}
            except RateLimitExceeded as e:
                yield {"event": "error", "data": json.dumps({"error": str(e), "session_id": session_id})}
                return
            except Exception as e:
                logger.exception("Stream failed")
                yield {"event": "error", "data": json.dumps({"error": str(e), "session_id": session_id})}
                return

            # Ensure clients always receive a done event so they can close cleanly.
            if not saw_done:
                yield {
                    "event": EVENT_DONE,
                    "data": json.dumps({"type": EVENT_DONE, "session_id": session_id, "answer": ""}),
                }
        finally:
            lock.release()

    return EventSourceResponse(event_stream())

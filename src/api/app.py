"""FastAPI app exposing the ResearchAgent as a REST + SSE service."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, query, sessions

# ─── Module overview ───────────────────────────────────────────────
# Builds the FastAPI application. The lifespan handler instantiates a
# singleton ResearchAgent at startup (expensive: opens the SQLite
# checkpointer, probes tool health) and stashes it on app.state along
# with a per-app asyncio.Lock that serializes mutating operations.
# Routers contribute /health, /query, /query/stream, and /sessions.
# ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# Builds the agent once at startup; tears down its DB connection on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the ResearchAgent at startup, close its resources at shutdown."""
    from src.agent import ResearchAgent

    logger.info("Initializing ResearchAgent...")
    app.state.agent = ResearchAgent()
    app.state.agent_lock = asyncio.Lock()
    logger.info(
        "Agent ready (%d tools enabled, %d disabled)",
        len(app.state.agent.tools),
        len(app.state.agent.disabled_tools),
    )

    try:
        yield
    finally:
        # Close the aiosqlite connection on the agent's owning IO loop so it
        # doesn't leak across event loops.
        agent = app.state.agent
        try:
            if hasattr(agent, "_db_conn") and agent._db_conn is not None:
                agent._io_loop.run(agent._db_conn.close())
        except Exception:
            logger.exception("Error closing checkpoint connection")


def create_app() -> FastAPI:
    """Factory that builds the FastAPI app. Used by uvicorn and tests."""
    app = FastAPI(
        title="Multi-Tool Research Agent",
        description=(
            "REST + SSE interface to the LangGraph research agent. "
            "Supports direct, plan-and-execute, and multi-agent modes."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS is intentionally permissive for local development. Tighten
    # `allow_origins` before deploying to a shared environment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(sessions.router)

    return app


app = create_app()

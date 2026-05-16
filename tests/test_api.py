"""Tests for src/api/* — REST endpoints over a mocked ResearchAgent."""

import asyncio
import json
import types
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from src.api.routes import health as health_route
from src.api.routes import query as query_route
from src.api.routes import sessions as sessions_route
from src.constants import EVENT_DONE, EVENT_STEP_TOOL, EVENT_SYNTHESIS_TOKEN


# ─────────────────────────────────────────────────────────────────────
# Fixtures: build a minimal FastAPI app whose app.state.agent is a stub.
# This avoids the cost (and side effects) of spinning up a real
# ResearchAgent — we just need to verify the route layer behaves.
# ─────────────────────────────────────────────────────────────────────


class StubMetrics:
    """Stand-in for ObservabilityCallbackHandler.get_metrics()."""

    total_tokens = 1234


class StubAgent:
    """Minimal ResearchAgent surface for route-layer testing."""

    def __init__(self):
        self.tools = [
            types.SimpleNamespace(name="calculator"),
            types.SimpleNamespace(name="web_search"),
        ]
        self.disabled_tools = []
        self.current_session_id = "stub_session"
        self.checkpointer = MagicMock()
        # Per-test scripts populated by tests; .query / .multi_agent_query
        # / .plan_and_execute are awaited so they must be coroutines.
        self.last_query = None
        self.last_mode_called = None

    def set_session_id(self, session_id: str) -> None:
        self.current_session_id = session_id

    async def query(self, q: str, show_timing: bool = False) -> str:
        self.last_query = q
        self.last_mode_called = "direct"
        return f"DIRECT: {q}"

    async def plan_and_execute(self, q: str, verbose: bool = False) -> str:
        self.last_query = q
        self.last_mode_called = "plan"
        return f"PLAN: {q}"

    async def multi_agent_query(self, q: str, verbose: bool = False) -> str:
        self.last_query = q
        self.last_mode_called = "multi"
        return f"MULTI: {q}"

    def get_last_metrics(self):
        return StubMetrics()

    def route_query(self, q: str, mode: str = "Auto"):
        # Sync generator matching the real agent's contract.
        yield {"type": EVENT_SYNTHESIS_TOKEN, "token": "Hello "}
        yield {"type": EVENT_SYNTHESIS_TOKEN, "token": "world."}
        yield {"type": EVENT_STEP_TOOL, "step_idx": -1, "tool_name": "web_search", "tool_output": "ok"}
        yield {"type": EVENT_DONE, "answer": f"STREAM: {q}"}


def _build_test_app(agent: StubAgent) -> FastAPI:
    """Build a FastAPI app preloaded with the stub agent (no lifespan)."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.agent = agent
    app.state.agent_lock = asyncio.Lock()
    app.include_router(health_route.router)
    app.include_router(query_route.router)
    app.include_router(sessions_route.router)
    return app


@pytest.fixture
def stub_agent() -> StubAgent:
    return StubAgent()


@pytest.fixture
def client(stub_agent: StubAgent) -> TestClient:
    return TestClient(_build_test_app(stub_agent))


# ─────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_status_ok_when_no_disabled_tools(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["tool_count"] == 2
        assert "calculator" in body["enabled_tools"]
        assert body["disabled_tools"] == []

    def test_status_degraded_when_tools_disabled(self, stub_agent: StubAgent):
        stub_agent.disabled_tools = ["wolfram_alpha"]
        client = TestClient(_build_test_app(stub_agent))

        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["disabled_tools"] == ["wolfram_alpha"]


# ─────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────


class TestQueryEndpoint:
    def test_direct_mode_returns_answer_and_metrics(self, client: TestClient):
        resp = client.post("/query", json={"query": "what is 2+2?", "mode": "direct"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "DIRECT: what is 2+2?"
        assert body["tokens_used"] == 1234
        assert body["session_id"]  # generated when not supplied
        assert body["duration_seconds"] >= 0

    def test_explicit_session_id_is_used(self, client: TestClient, stub_agent: StubAgent):
        resp = client.post(
            "/query",
            json={"query": "hi", "mode": "direct", "session_id": "custom_thread_42"},
        )
        assert resp.json()["session_id"] == "custom_thread_42"
        assert stub_agent.current_session_id == "custom_thread_42"

    def test_plan_mode_dispatches_to_plan_method(self, client: TestClient, stub_agent: StubAgent):
        resp = client.post("/query", json={"query": "complex topic", "mode": "plan"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "PLAN: complex topic"
        assert stub_agent.last_mode_called == "plan"

    def test_multi_mode_dispatches_to_multi_agent(self, client: TestClient, stub_agent: StubAgent):
        resp = client.post("/query", json={"query": "compare X and Y", "mode": "multi"})
        assert resp.json()["answer"] == "MULTI: compare X and Y"
        assert stub_agent.last_mode_called == "multi"

    def test_invalid_mode_rejected_by_schema(self, client: TestClient):
        resp = client.post("/query", json={"query": "x", "mode": "unknown"})
        assert resp.status_code == 422  # Pydantic validation

    def test_empty_query_rejected_by_schema(self, client: TestClient):
        resp = client.post("/query", json={"query": "", "mode": "direct"})
        assert resp.status_code == 422

    def test_rate_limit_exceeded_becomes_429(self, client: TestClient, stub_agent: StubAgent):
        from src.rate_limiter import RateLimitExceeded

        async def boom(q, show_timing=False):
            raise RateLimitExceeded(tokens_spent=200_000, budget=100_000)

        stub_agent.query = boom
        resp = client.post("/query", json={"query": "anything", "mode": "direct"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]

    def test_generic_error_becomes_500(self, client: TestClient, stub_agent: StubAgent):
        async def boom(q, show_timing=False):
            raise RuntimeError("backend offline")

        stub_agent.query = boom
        resp = client.post("/query", json={"query": "anything", "mode": "direct"})
        assert resp.status_code == 500
        assert "backend offline" in resp.json()["detail"]


# ─────────────────────────────────────────────────────────────────────
# POST /query/stream  (SSE)
# ─────────────────────────────────────────────────────────────────────


def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE wire format into a list of {event, data: <dict>} entries.

    sse-starlette uses CRLF line terminators and CRLF CRLF event separators,
    so normalize to LF first to keep the split logic readable.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    events = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue  # SSE comment / keepalive
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            payload = json.loads("\n".join(data_lines))
            events.append({"event": event_name, "data": payload})
    return events


class TestQueryStreamEndpoint:
    def test_streams_typed_events_terminated_by_done(self, client: TestClient):
        with client.stream("POST", "/query/stream", json={"query": "hello", "mode": "direct"}) as r:
            assert r.status_code == 200
            body = "".join(chunk for chunk in r.iter_text())

        events = _parse_sse(body)
        types_seen = [e["event"] for e in events]
        assert EVENT_SYNTHESIS_TOKEN in types_seen
        assert EVENT_STEP_TOOL in types_seen
        assert types_seen[-1] == EVENT_DONE
        # session_id is threaded into every event payload
        assert all(e["data"].get("session_id") for e in events)
        # The done event carries the final answer
        assert events[-1]["data"]["answer"] == "STREAM: hello"

    def test_uses_caller_supplied_session_id(self, client: TestClient):
        with client.stream(
            "POST",
            "/query/stream",
            json={"query": "hi", "mode": "direct", "session_id": "thread_xyz"},
        ) as r:
            body = "".join(chunk for chunk in r.iter_text())

        events = _parse_sse(body)
        for e in events:
            assert e["data"]["session_id"] == "thread_xyz"

    def test_stream_error_becomes_error_event(self, client: TestClient, stub_agent: StubAgent):
        def boom_gen(q, mode="Auto"):
            yield {"type": EVENT_SYNTHESIS_TOKEN, "token": "starting "}
            raise RuntimeError("upstream broke")

        stub_agent.route_query = boom_gen

        with client.stream("POST", "/query/stream", json={"query": "hi", "mode": "direct"}) as r:
            assert r.status_code == 200
            body = "".join(chunk for chunk in r.iter_text())

        events = _parse_sse(body)
        assert events[-1]["event"] == "error"
        assert "upstream broke" in events[-1]["data"]["error"]


# ─────────────────────────────────────────────────────────────────────
# /sessions
# ─────────────────────────────────────────────────────────────────────


class TestSessions:
    def test_list_returns_summaries(self, client: TestClient, stub_agent: StubAgent, monkeypatch):
        monkeypatch.setattr(
            sessions_route, "list_sessions",
            lambda cp: [
                {"session_id": "s1", "created_at": "2026-05-16", "message_count": 4},
                {"session_id": "s2", "created_at": "2026-05-15", "message_count": 2},
            ],
        )

        resp = client.get("/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["session_id"] == "s1"
        assert body[0]["message_count"] == 4

    def test_get_session_returns_exchanges(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(
            sessions_route, "load_session",
            lambda cp, sid: [("hi", "hello"), ("how are you", "great")],
        )

        resp = client.get("/sessions/abc")
        body = resp.json()
        assert resp.status_code == 200
        assert body["session_id"] == "abc"
        assert body["exchanges"][0] == {"question": "hi", "answer": "hello"}
        assert len(body["exchanges"]) == 2

    def test_get_missing_session_returns_404(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(sessions_route, "load_session", lambda cp, sid: None)
        resp = client.get("/sessions/missing")
        assert resp.status_code == 404

    def test_delete_session_success(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(sessions_route, "delete_session", lambda cp, sid: True)
        resp = client.delete("/sessions/abc")
        assert resp.status_code == 200
        assert resp.json() == {"session_id": "abc", "deleted": True}

    def test_delete_missing_session_returns_404(self, client: TestClient, monkeypatch):
        monkeypatch.setattr(sessions_route, "delete_session", lambda cp, sid: False)
        resp = client.delete("/sessions/missing")
        assert resp.status_code == 404

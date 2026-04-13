"""Tests for session management and SqliteSaver checkpointing integration."""

import sqlite3
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from src.session_manager import (
    list_sessions,
    load_session,
    delete_session,
    get_session_preview,
    generate_session_id,
)


class TestCheckpointerSetup:
    """Test that SqliteSaver initializes correctly."""

    def test_setup_creates_tables(self, checkpointer):
        cursor = checkpointer.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "checkpoints" in tables
        assert "writes" in tables

    def test_empty_list(self, checkpointer):
        results = list(checkpointer.list(None))
        assert results == []

    def test_get_tuple_missing_thread(self, checkpointer):
        config = {"configurable": {"thread_id": "nonexistent"}}
        result = checkpointer.get_tuple(config)
        assert result is None


@pytest.fixture
def checkpointer():
    """In-memory SqliteSaver for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn=conn)
    saver.setup()
    yield saver
    conn.close()


def _put_checkpoint(checkpointer, thread_id, messages):
    """Helper: store a checkpoint with the given messages for a thread."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": ""}}
    checkpoint = {
        "v": 1,
        "id": thread_id,
        "ts": "2026-04-13T12:00:00+00:00",
        "channel_values": {"messages": messages},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {}
    checkpointer.put(config, checkpoint, metadata, {})


class TestGenerateSessionId:
    """Test session ID generation."""

    async def test_with_description(self):
        sid = generate_session_id("Tesla stock research")
        assert "tesla" in sid
        assert "stock" in sid
        assert "research" in sid

    async def test_limits_to_three_words(self):
        sid = generate_session_id("one two three four five")
        parts = sid.split("_")
        assert len(parts) <= 4

    async def test_without_description(self):
        sid = generate_session_id()
        assert "session" in sid

    async def test_cleans_special_characters(self):
        sid = generate_session_id("test!@# data")
        assert "!" not in sid
        assert "@" not in sid


class TestListSessions:
    """Test listing sessions from checkpoints."""

    async def test_list_empty(self, checkpointer):
        sessions = list_sessions(checkpointer)
        assert sessions == []

    async def test_list_returns_saved_sessions(self, checkpointer):
        _put_checkpoint(checkpointer, "session_one", [
            HumanMessage(content="Q1"), AIMessage(content="A1"),
        ])
        _put_checkpoint(checkpointer, "session_two", [
            HumanMessage(content="Q2"), AIMessage(content="A2"),
        ])

        sessions = list_sessions(checkpointer)
        assert len(sessions) == 2
        ids = [s["session_id"] for s in sessions]
        assert "session_one" in ids
        assert "session_two" in ids

    async def test_list_skips_step_threads(self, checkpointer):
        _put_checkpoint(checkpointer, "main_session", [
            HumanMessage(content="Q1"), AIMessage(content="A1"),
        ])
        _put_checkpoint(checkpointer, "main_session__step_1", [
            HumanMessage(content="step task"),
        ])

        sessions = list_sessions(checkpointer)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "main_session"


class TestLoadSession:
    """Test loading sessions from checkpoints."""

    async def test_load_restores_history(self, checkpointer):
        _put_checkpoint(checkpointer, "load_test", [
            HumanMessage(content="Q1"), AIMessage(content="A1"),
            HumanMessage(content="Q2"), AIMessage(content="A2"),
        ])

        loaded = load_session(checkpointer, "load_test")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0] == ("Q1", "A1")
        assert loaded[1] == ("Q2", "A2")

    async def test_load_nonexistent_returns_none(self, checkpointer):
        result = load_session(checkpointer, "nonexistent_session")
        assert result is None


class TestDeleteSession:
    """Test deleting sessions."""

    async def test_delete_existing(self, checkpointer):
        _put_checkpoint(checkpointer, "to_delete", [
            HumanMessage(content="Q1"), AIMessage(content="A1"),
        ])
        result = delete_session(checkpointer, "to_delete")
        assert result is True

        # Verify it's gone
        assert load_session(checkpointer, "to_delete") is None

    async def test_delete_nonexistent(self, checkpointer):
        result = delete_session(checkpointer, "does_not_exist")
        assert result is False


class TestSessionPreview:
    """Test session preview generation."""

    async def test_preview_shows_exchanges(self, checkpointer):
        _put_checkpoint(checkpointer, "preview_test", [
            HumanMessage(content="What is AI?"), AIMessage(content="AI is..."),
        ])
        preview = get_session_preview(checkpointer, "preview_test")
        assert "What is AI?" in preview
        assert "AI is..." in preview

    async def test_preview_nonexistent(self, checkpointer):
        result = get_session_preview(checkpointer, "nonexistent")
        assert result is None

    async def test_preview_truncates_long_messages(self, checkpointer):
        long_msg = "A" * 100
        _put_checkpoint(checkpointer, "long_test", [
            HumanMessage(content=long_msg), AIMessage(content="short"),
        ])
        preview = get_session_preview(checkpointer, "long_test")
        assert "..." in preview

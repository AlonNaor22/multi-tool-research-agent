"""Tests for SqliteSaver checkpointing integration in ResearchAgent.

SimpleMemory was replaced by SqliteSaver — these tests verify the
new checkpointing-based conversation persistence.
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver


@pytest.fixture
def checkpointer():
    """In-memory SqliteSaver for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn=conn)
    saver.setup()
    yield saver
    conn.close()


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


class TestGenerateSessionId:
    """Test session ID generation (unchanged from old system)."""

    def test_with_description(self):
        from src.session_manager import generate_session_id
        sid = generate_session_id("Tesla stock research")
        assert "tesla" in sid
        assert "stock" in sid

    def test_without_description(self):
        from src.session_manager import generate_session_id
        sid = generate_session_id()
        assert "session" in sid

    def test_cleans_special_characters(self):
        from src.session_manager import generate_session_id
        sid = generate_session_id("test!@# data")
        assert "!" not in sid
        assert "@" not in sid

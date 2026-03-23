"""Tests for src/session_manager.py — save/load/list sessions."""

import os
import json
import pytest
from unittest.mock import patch
from src.session_manager import (
    save_session,
    load_session,
    list_sessions,
    delete_session,
    get_session_preview,
    generate_session_id,
    SESSIONS_DIR,
)


@pytest.fixture
def sessions_dir(tmp_path):
    """Use a temporary directory for session files."""
    test_dir = str(tmp_path / "sessions")
    with patch("src.session_manager.SESSIONS_DIR", test_dir):
        os.makedirs(test_dir, exist_ok=True)
        yield test_dir


class TestGenerateSessionId:
    """Test session ID generation."""

    def test_with_description(self):
        sid = generate_session_id("Tesla stock research")
        assert "tesla" in sid
        assert "stock" in sid
        assert "research" in sid

    def test_limits_to_three_words(self):
        sid = generate_session_id("one two three four five")
        parts = sid.split("_")
        # Date + 3 words max
        assert len(parts) <= 4

    def test_without_description(self):
        sid = generate_session_id()
        assert "session" in sid

    def test_cleans_special_characters(self):
        sid = generate_session_id("test!@# data")
        assert "!" not in sid
        assert "@" not in sid


class TestSaveSession:
    """Test saving sessions."""

    def test_save_creates_file(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            history = [("What is AI?", "AI is artificial intelligence.")]
            filepath = save_session(history, session_id="test_session")

            assert os.path.exists(filepath)

    def test_save_correct_json_structure(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            history = [("Q1", "A1"), ("Q2", "A2")]
            filepath = save_session(history, session_id="test_session")

            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["session_id"] == "test_session"
            assert data["message_count"] == 2
            assert len(data["history"]) == 2
            assert data["history"][0]["input"] == "Q1"
            assert data["history"][0]["output"] == "A1"

    def test_save_with_description(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            history = [("Q1", "A1")]
            filepath = save_session(history, description="my research")
            assert os.path.exists(filepath)


class TestLoadSession:
    """Test loading sessions."""

    def test_load_restores_history(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            history = [("Q1", "A1"), ("Q2", "A2")]
            save_session(history, session_id="load_test")

            loaded = load_session("load_test")
            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0] == ("Q1", "A1")
            assert loaded[1] == ("Q2", "A2")

    def test_load_nonexistent_returns_none(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            result = load_session("nonexistent_session")
            assert result is None

    def test_load_corrupted_json(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            filepath = os.path.join(sessions_dir, "corrupted.json")
            with open(filepath, "w") as f:
                f.write("not valid json {{{")

            result = load_session("corrupted")
            assert result is None


class TestListSessions:
    """Test listing sessions."""

    def test_list_empty(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            sessions = list_sessions()
            assert sessions == []

    def test_list_returns_saved_sessions(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            save_session([("Q1", "A1")], session_id="session_one")
            save_session([("Q2", "A2")], session_id="session_two")

            sessions = list_sessions()
            assert len(sessions) == 2
            ids = [s["session_id"] for s in sessions]
            assert "session_one" in ids
            assert "session_two" in ids

    def test_list_skips_corrupted_files(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            save_session([("Q1", "A1")], session_id="good_session")

            # Create a corrupted file
            with open(os.path.join(sessions_dir, "bad.json"), "w") as f:
                f.write("corrupted")

            sessions = list_sessions()
            assert len(sessions) == 1


class TestDeleteSession:
    """Test deleting sessions."""

    def test_delete_existing(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            save_session([("Q1", "A1")], session_id="to_delete")
            result = delete_session("to_delete")
            assert result is True
            assert not os.path.exists(os.path.join(sessions_dir, "to_delete.json"))

    def test_delete_nonexistent(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            result = delete_session("does_not_exist")
            assert result is False


class TestSessionPreview:
    """Test session preview generation."""

    def test_preview_shows_exchanges(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            save_session([("What is AI?", "AI is...")], session_id="preview_test")
            preview = get_session_preview("preview_test")
            assert "What is AI?" in preview
            assert "AI is..." in preview

    def test_preview_nonexistent(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            result = get_session_preview("nonexistent")
            assert result is None

    def test_preview_truncates_long_messages(self, sessions_dir):
        with patch("src.session_manager.SESSIONS_DIR", sessions_dir):
            long_msg = "A" * 100
            save_session([(long_msg, "short")], session_id="long_test")
            preview = get_session_preview("long_test")
            assert "..." in preview

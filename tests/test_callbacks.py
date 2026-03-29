"""Tests for src/callbacks.py — timing callback handler."""

import time
import pytest
from src.callbacks import TimingCallbackHandler


@pytest.fixture
def handler():
    """Fresh callback handler."""
    return TimingCallbackHandler()


class TestToolTracking:
    """Test that tool execution is tracked correctly."""

    async def test_tool_start_and_end(self, handler):
        handler.on_tool_start({"name": "calculator"}, "2 + 2")
        handler.on_tool_end("4")

        assert len(handler.tool_times) == 1
        assert handler.tool_times[0]["tool"] == "calculator"
        assert handler.tool_times[0]["duration"] >= 0

    async def test_multiple_tools(self, handler):
        handler.on_tool_start({"name": "calculator"}, "2 + 2")
        handler.on_tool_end("4")

        handler.on_tool_start({"name": "web_search"}, "AI news")
        handler.on_tool_end("results...")

        assert len(handler.tool_times) == 2
        assert handler.tool_times[0]["tool"] == "calculator"
        assert handler.tool_times[1]["tool"] == "web_search"

    async def test_unknown_tool_name(self, handler):
        handler.on_tool_start({}, "input")
        handler.on_tool_end("output")

        assert handler.tool_times[0]["tool"] == "unknown_tool"


class TestToolError:
    """Test error tracking."""

    async def test_error_resets_state(self, handler):
        handler.on_tool_start({"name": "weather"}, "London")
        handler.on_tool_error(Exception("API timeout"))

        # State should be reset so next tool works
        assert handler._current_tool_start is None
        assert handler._current_tool_name is None


class TestSummary:
    """Test summary generation."""

    async def test_empty_summary(self, handler):
        assert handler.get_summary() == "No tools were used."

    async def test_summary_with_tools(self, handler):
        handler.on_tool_start({"name": "calculator"}, "2+2")
        handler.on_tool_end("4")

        summary = handler.get_summary()
        assert "calculator" in summary
        assert "Total tool time" in summary

    async def test_summary_shows_all_tools(self, handler):
        for name in ["calculator", "web_search", "wikipedia"]:
            handler.on_tool_start({"name": name}, "input")
            handler.on_tool_end("output")

        summary = handler.get_summary()
        assert "calculator" in summary
        assert "web_search" in summary
        assert "wikipedia" in summary


class TestReset:
    """Test reset functionality."""

    async def test_reset_clears_data(self, handler):
        handler.on_tool_start({"name": "calculator"}, "2+2")
        handler.on_tool_end("4")
        handler.reset()

        assert len(handler.tool_times) == 0
        assert handler._current_tool_start is None
        assert handler._current_tool_name is None
        assert handler.get_summary() == "No tools were used."

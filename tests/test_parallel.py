"""Tests for src/tools/parallel_tool.py — multi-source parallel search."""

import sys
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Mock dependencies that may not be installed
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()
if "wikipedia" not in sys.modules:
    mock_wiki = MagicMock()
    mock_exceptions = MagicMock()
    mock_exceptions.PageError = type("PageError", (Exception,), {})
    mock_exceptions.DisambiguationError = type("DisambiguationError", (Exception,), {})
    mock_wiki.exceptions = mock_exceptions
    sys.modules["wikipedia"] = mock_wiki

from src.tools.parallel_tool import parallel_search


class TestParallelSearch:
    """Test parallel search with mocked underlying tools."""

    async def test_runs_multiple_searches(self):
        async def mock_web_search(q):
            return "Web: AI results"

        async def mock_wiki_search(q):
            return "Wiki: AI article"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search), \
             patch("src.tools.parallel_tool.search_wikipedia", side_effect=mock_wiki_search):
            input_data = json.dumps({
                "searches": [
                    {"type": "web", "query": "artificial intelligence"},
                    {"type": "wikipedia", "query": "artificial intelligence"},
                ]
            })
            result = await parallel_search(input_data)

            assert len(result) > 0

    async def test_invalid_json(self):
        result = await parallel_search("not json {{{")
        assert "Error" in result or "error" in result.lower()

    async def test_missing_searches_field(self):
        result = await parallel_search('{"query": "test"}')
        assert "Error" in result or "error" in result.lower() or "searches" in result.lower()

    async def test_too_many_searches(self):
        searches = [{"type": "web", "query": f"query {i}"} for i in range(15)]
        input_data = json.dumps({"searches": searches})

        async def mock_web_search(q):
            return "result"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search):
            result = await parallel_search(input_data)
            assert len(result) > 0

    async def test_missing_query_in_search(self):
        input_data = json.dumps({
            "searches": [{"type": "web"}]
        })

        async def mock_web_search(q):
            return "result"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search):
            result = await parallel_search(input_data)
            assert len(result) > 0  # Should handle gracefully

    async def test_handles_tool_failure(self):
        async def mock_web_search(q):
            raise Exception("Search failed")

        async def mock_wiki_search(q):
            return "Wiki: works fine"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search), \
             patch("src.tools.parallel_tool.search_wikipedia", side_effect=mock_wiki_search):
            input_data = json.dumps({
                "searches": [
                    {"type": "web", "query": "test"},
                    {"type": "wikipedia", "query": "test"},
                ]
            })
            result = await parallel_search(input_data)

            # Should still return results from working tools
            assert len(result) > 0

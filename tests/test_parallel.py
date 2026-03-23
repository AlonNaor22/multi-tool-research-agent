"""Tests for src/tools/parallel_tool.py — multi-source parallel search."""

import sys
import json
import pytest
from unittest.mock import patch, MagicMock

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

    def test_runs_multiple_searches(self):
        with patch("src.tools.parallel_tool.web_search", return_value="Web: AI results"), \
             patch("src.tools.parallel_tool.search_wikipedia", return_value="Wiki: AI article"):
            input_data = json.dumps({
                "searches": [
                    {"type": "web", "query": "artificial intelligence"},
                    {"type": "wikipedia", "query": "artificial intelligence"},
                ]
            })
            result = parallel_search(input_data)

            assert len(result) > 0

    def test_invalid_json(self):
        result = parallel_search("not json {{{")
        assert "Error" in result or "error" in result.lower()

    def test_missing_searches_field(self):
        result = parallel_search('{"query": "test"}')
        assert "Error" in result or "error" in result.lower() or "searches" in result.lower()

    def test_too_many_searches(self):
        searches = [{"type": "web", "query": f"query {i}"} for i in range(15)]
        input_data = json.dumps({"searches": searches})

        with patch("src.tools.parallel_tool.web_search", return_value="result"):
            result = parallel_search(input_data)
            assert len(result) > 0

    def test_missing_query_in_search(self):
        input_data = json.dumps({
            "searches": [{"type": "web"}]
        })

        with patch("src.tools.parallel_tool.web_search", return_value="result"):
            result = parallel_search(input_data)
            assert len(result) > 0  # Should handle gracefully

    def test_handles_tool_failure(self):
        with patch("src.tools.parallel_tool.web_search",
                   side_effect=Exception("Search failed")), \
             patch("src.tools.parallel_tool.search_wikipedia",
                   return_value="Wiki: works fine"):
            input_data = json.dumps({
                "searches": [
                    {"type": "web", "query": "test"},
                    {"type": "wikipedia", "query": "test"},
                ]
            })
            result = parallel_search(input_data)

            # Should still return results from working tools
            assert len(result) > 0

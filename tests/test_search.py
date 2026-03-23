"""Tests for src/tools/search_tool.py — DuckDuckGo web search."""

import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock duckduckgo_search before importing the tool
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()

from src.tools.search_tool import web_search


class TestWebSearch:
    """Test web search functionality with mocked DuckDuckGo API."""

    def test_returns_formatted_results(self, search_results):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = search_results
            mock_ddgs_cls.return_value = mock_instance

            result = web_search("test query")

            assert "Test Result 1" in result
            assert "Test Result 2" in result
            assert "example.com/1" in result
            assert "Found 2 results" in result

    def test_no_results(self):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs_cls.return_value = mock_instance

            result = web_search("obscure query no results")

            assert "No search results" in result

    def test_json_input_with_options(self, search_results):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = search_results
            mock_ddgs_cls.return_value = mock_instance

            result = web_search('{"query": "AI news", "max_results": 3}')

            assert "AI news" in result
            call_kwargs = mock_instance.text.call_args[1]
            assert call_kwargs["max_results"] == 3

    def test_empty_query(self):
        result = web_search("")
        assert "Error" in result or "No search query" in result

    def test_truncates_long_snippets(self):
        long_result = [{
            "title": "Long Result",
            "href": "https://example.com",
            "body": "A" * 500,
        }]
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = long_result
            mock_ddgs_cls.return_value = mock_instance

            result = web_search("test")

            assert "..." in result

    def test_handles_api_error(self):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = Exception("Rate limited")
            mock_ddgs_cls.return_value = mock_instance

            result = web_search("test query")

            assert "Error" in result

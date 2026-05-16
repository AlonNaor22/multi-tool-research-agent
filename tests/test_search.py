"""Tests for src/tools/search_tool.py — DuckDuckGo web search."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# Mock duckduckgo_search before importing the tool
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()

from src.tools.search_tool import web_search, search_tool, WebSearchInput


class TestWebSearch:
    """Test web search functionality with mocked DuckDuckGo API."""

    async def test_returns_formatted_results(self, search_results):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = search_results
            mock_ddgs_cls.return_value = mock_instance

            result = await web_search("test query")

            assert "Test Result 1" in result
            assert "Test Result 2" in result
            assert "example.com/1" in result
            assert "Found 2 results" in result

    async def test_no_results(self):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs_cls.return_value = mock_instance

            result = await web_search("obscure query no results")

            assert "No search results" in result

    async def test_max_results_kwarg(self, search_results):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = search_results
            mock_ddgs_cls.return_value = mock_instance

            await web_search("AI news", max_results=3)

            call_kwargs = mock_instance.text.call_args[1]
            assert call_kwargs["max_results"] == 3

    async def test_region_kwarg(self, search_results):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = search_results
            mock_ddgs_cls.return_value = mock_instance

            await web_search("local news", region="uk-en")

            call_kwargs = mock_instance.text.call_args[1]
            assert call_kwargs["region"] == "uk-en"

    async def test_empty_query(self):
        result = await web_search("")
        assert "Error" in result or "No search query" in result

    async def test_truncates_long_snippets(self):
        long_result = [{
            "title": "Long Result",
            "href": "https://example.com",
            "body": "A" * 500,
        }]
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.return_value = long_result
            mock_ddgs_cls.return_value = mock_instance

            result = await web_search("test")

            assert "..." in result

    async def test_handles_api_error(self):
        with patch("src.tools.search_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = Exception("Rate limited")
            mock_ddgs_cls.return_value = mock_instance

            result = await web_search("test query")

            assert "Error" in result


class TestWebSearchSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        with pytest.raises(ValidationError):
            WebSearchInput()

    def test_max_results_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            WebSearchInput(query="test", max_results=0)
        with pytest.raises(ValidationError):
            WebSearchInput(query="test", max_results=100)

    def test_valid_input_parses(self):
        parsed = WebSearchInput(query="hello", max_results=3, region="us-en")
        assert parsed.query == "hello"
        assert parsed.max_results == 3
        assert parsed.region == "us-en"


class TestSearchTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        assert search_tool.name == "web_search"
        assert search_tool.args_schema is WebSearchInput

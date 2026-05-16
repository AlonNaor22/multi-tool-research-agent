"""Tests for src/tools/news_tool.py — news article search."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# Mock duckduckgo_search before importing the tool
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()

from src.tools.news_tool import news_search, news_tool, NewsSearchInput


class TestNewsSearch:
    """Test news search with mocked DuckDuckGo News API.

    Each test patches `src.tools.news_tool.DDGS` locally (rather than
    mutating sys.modules['duckduckgo_search']) so news tests don't fight
    over the shared mock with test_search.py and test_parallel.py under
    pytest-xdist parallel execution.
    """

    async def test_returns_formatted_results(self):
        mock_results = [
            {
                "title": "AI Breakthrough",
                "url": "https://news.example.com/1",
                "body": "Scientists discover new AI technique.",
                "date": "2026-03-20",
                "source": "Tech News",
            },
            {
                "title": "Market Update",
                "url": "https://news.example.com/2",
                "body": "Stock markets rally on strong earnings.",
                "date": "2026-03-19",
                "source": "Finance Daily",
            },
        ]

        with patch("src.tools.news_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.news.return_value = mock_results
            mock_ddgs_cls.return_value = mock_instance

            result = await news_search("AI news")

            assert "AI Breakthrough" in result
            assert "Market Update" in result

    async def test_no_results(self):
        with patch("src.tools.news_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.news.return_value = []
            mock_ddgs_cls.return_value = mock_instance

            result = await news_search("obscure topic no results")

            assert "No news" in result or "no" in result.lower()

    async def test_timelimit_kwarg(self):
        mock_results = [{
            "title": "Recent News",
            "url": "https://example.com",
            "body": "Something happened.",
            "date": "2026-03-22",
            "source": "Source",
        }]

        with patch("src.tools.news_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.news.return_value = mock_results
            mock_ddgs_cls.return_value = mock_instance

            result = await news_search("tech", timelimit="d")

            assert "Recent News" in result
            call_kwargs = mock_instance.news.call_args[1]
            assert call_kwargs["timelimit"] == "d"

    async def test_empty_query(self):
        result = await news_search("")
        assert "Error" in result or "No" in result

    async def test_handles_api_error(self):
        with patch("src.tools.news_tool.DDGS") as mock_ddgs_cls:
            mock_instance = MagicMock()
            mock_instance.news.side_effect = Exception("API error")
            mock_ddgs_cls.return_value = mock_instance

            result = await news_search("test")

            assert "Error" in result or "error" in result.lower()


class TestNewsSearchSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        with pytest.raises(ValidationError):
            NewsSearchInput()

    def test_invalid_timelimit_rejected(self):
        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", timelimit="year")

    def test_max_results_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", max_results=0)

    def test_valid_input_parses(self):
        parsed = NewsSearchInput(query="climate", timelimit="m", max_results=3)
        assert parsed.timelimit == "m"
        assert parsed.max_results == 3


class TestNewsTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        assert news_tool.name == "news_search"
        assert news_tool.args_schema is NewsSearchInput

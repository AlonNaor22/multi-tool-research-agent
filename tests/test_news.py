"""Tests for src/tools/news_tool.py — news article search."""

import sys
import pytest
from unittest.mock import MagicMock

# Mock duckduckgo_search before importing the tool
if "duckduckgo_search" not in sys.modules:
    sys.modules["duckduckgo_search"] = MagicMock()

mock_ddgs = sys.modules["duckduckgo_search"]

from src.tools.news_tool import search_news


class TestNewsSearch:
    """Test news search with mocked DuckDuckGo News API."""

    def test_returns_formatted_results(self):
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

        mock_instance = MagicMock()
        mock_instance.news.return_value = mock_results
        mock_ddgs.DDGS.return_value = mock_instance

        result = search_news("AI news")

        assert "AI Breakthrough" in result
        assert "Market Update" in result

    def test_no_results(self):
        mock_instance = MagicMock()
        mock_instance.news.return_value = []
        mock_ddgs.DDGS.return_value = mock_instance

        result = search_news("obscure topic no results")

        assert "No news" in result or "no" in result.lower()

    def test_json_input_with_timelimit(self):
        mock_results = [{
            "title": "Recent News",
            "url": "https://example.com",
            "body": "Something happened.",
            "date": "2026-03-22",
            "source": "Source",
        }]

        mock_instance = MagicMock()
        mock_instance.news.return_value = mock_results
        mock_ddgs.DDGS.return_value = mock_instance

        result = search_news('{"query": "tech", "timelimit": "d"}')

        assert "Recent News" in result

    def test_empty_query(self):
        result = search_news("")
        assert len(result) > 0

    def test_handles_api_error(self):
        mock_instance = MagicMock()
        mock_instance.news.side_effect = Exception("API error")
        mock_ddgs.DDGS.return_value = mock_instance

        result = search_news("test")

        assert "Error" in result or "error" in result.lower()

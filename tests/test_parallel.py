"""Tests for src/tools/parallel_tool.py — multi-source parallel search."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

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

from src.tools.parallel_tool import (
    parallel_search,
    parallel_tool,
    ParallelSearchInput,
    SearchSpec,
)


class TestParallelSearch:
    """Test parallel search with mocked underlying tools."""

    async def test_runs_multiple_searches(self):
        async def mock_web_search(q):
            return "Web: AI results"

        async def mock_wiki_search(q):
            return "Wiki: AI article"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search), \
             patch("src.tools.parallel_tool.wikipedia", side_effect=mock_wiki_search):
            searches = [
                {"type": "web", "query": "artificial intelligence"},
                {"type": "wikipedia", "query": "artificial intelligence"},
            ]
            result = await parallel_search(searches)
            assert "2/2 successful" in result
            assert "Web: AI results" in result
            assert "Wiki: AI article" in result

    async def test_handles_tool_failure(self):
        async def mock_web_search(q):
            raise Exception("Search failed")

        async def mock_wiki_search(q):
            return "Wiki: works fine"

        with patch("src.tools.parallel_tool.web_search", side_effect=mock_web_search), \
             patch("src.tools.parallel_tool.wikipedia", side_effect=mock_wiki_search):
            searches = [
                {"type": "web", "query": "test"},
                {"type": "wikipedia", "query": "test"},
            ]
            result = await parallel_search(searches)

            # The wikipedia branch still succeeds; the web branch surfaces as FAILED.
            assert "Wiki: works fine" in result
            assert "FAILED" in result
            assert "Search failed" in result


class TestParallelSearchSchema:
    """Pydantic args_schema enforces shape; LangChain rejects bad calls at the boundary."""

    def test_missing_searches_field_rejected(self):
        with pytest.raises(ValidationError):
            ParallelSearchInput()

    def test_empty_searches_rejected(self):
        with pytest.raises(ValidationError):
            ParallelSearchInput(searches=[])

    def test_too_many_searches_rejected(self):
        too_many = [{"type": "web", "query": f"q{i}"} for i in range(11)]
        with pytest.raises(ValidationError):
            ParallelSearchInput(searches=too_many)

    def test_missing_query_in_search_rejected(self):
        with pytest.raises(ValidationError):
            SearchSpec(type="web")

    def test_unknown_search_type_rejected(self):
        with pytest.raises(ValidationError):
            SearchSpec(type="bing", query="test")

    def test_valid_input_parses(self):
        parsed = ParallelSearchInput(
            searches=[{"type": "web", "query": "hello"}]
        )
        assert parsed.searches[0].type == "web"
        assert parsed.searches[0].query == "hello"


class TestParallelTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        assert parallel_tool.name == "parallel_search"
        assert parallel_tool.args_schema is ParallelSearchInput

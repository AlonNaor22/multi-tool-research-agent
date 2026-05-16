"""Tests for src/tools/wikipedia_tool.py — Wikipedia lookups."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# Mock the wikipedia module if not installed
if "wikipedia" not in sys.modules:
    mock_wiki = MagicMock()
    # Create proper exception classes
    mock_exceptions = MagicMock()
    mock_exceptions.PageError = type("PageError", (Exception,), {})
    mock_exceptions.DisambiguationError = type("DisambiguationError", (Exception,), {
        "__init__": lambda self, title, may_refer_to: (
            setattr(self, "title", title) or setattr(self, "options", may_refer_to)
        ),
    })
    mock_wiki.exceptions = mock_exceptions
    sys.modules["wikipedia"] = mock_wiki

import wikipedia as wiki_module
from src.tools.wikipedia_tool import wikipedia, wikipedia_tool, WikipediaInput


class TestWikipediaSearch:
    """Test Wikipedia search with mocked API."""

    async def test_returns_summary(self):
        wiki_module.summary = MagicMock(return_value="Python is a programming language.")
        result = await wikipedia("Python programming")
        assert "Python" in result
        assert "programming language" in result

    async def test_page_not_found(self):
        wiki_module.summary = MagicMock(
            side_effect=wiki_module.exceptions.PageError("test")
        )
        result = await wikipedia("xyznonexistentpage123")
        assert len(result) > 0  # Should return an error message, not crash

    async def test_disambiguation_handling(self):
        error = wiki_module.exceptions.DisambiguationError(
            "Python", ["Python (programming)", "Python (snake)"]
        )
        wiki_module.summary = MagicMock(side_effect=error)
        result = await wikipedia("Python")
        assert len(result) > 0  # Should handle gracefully

    async def test_sentences_kwarg(self):
        wiki_module.summary = MagicMock(return_value="Test summary.")
        result = await wikipedia("Python", sentences=2)
        assert "Test summary" in result
        # Verify sentences was passed through
        call_kwargs = wiki_module.summary.call_args[1]
        assert call_kwargs.get("sentences") == 2


class TestWikipediaSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        with pytest.raises(ValidationError):
            WikipediaInput()

    def test_sentences_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            WikipediaInput(query="test", sentences=0)
        with pytest.raises(ValidationError):
            WikipediaInput(query="test", sentences=21)

    def test_valid_input_parses(self):
        parsed = WikipediaInput(query="Python", sentences=3, suggestion=False, results=2)
        assert parsed.query == "Python"
        assert parsed.sentences == 3
        assert parsed.suggestion is False
        assert parsed.results == 2


class TestWikipediaTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        assert wikipedia_tool.name == "wikipedia"
        assert wikipedia_tool.args_schema is WikipediaInput

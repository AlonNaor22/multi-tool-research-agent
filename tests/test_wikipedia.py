"""Tests for src/tools/wikipedia_tool.py — Wikipedia lookups."""

import sys
import pytest
from unittest.mock import patch, MagicMock

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
from src.tools.wikipedia_tool import wikipedia


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

    async def test_json_input_with_options(self):
        wiki_module.summary = MagicMock(return_value="Test summary.")
        result = await wikipedia('{"query": "Python", "sentences": 2}')
        assert "Test summary" in result

"""Tests for src/tools/arxiv_tool.py — academic paper search."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError


class MockArxivResult:
    """Mock arxiv search result."""

    def __init__(self, title, authors, summary, published, entry_id, categories=None):
        self.title = title
        self.authors = [MagicMock(name=a) for a in authors]
        # Set the name attribute properly for each mock author
        for author, name in zip(self.authors, authors):
            author.name = name
        self.summary = summary
        self.published = published
        self.entry_id = entry_id
        self.categories = categories or ["cs.AI"]


def _patch_arxiv(mock_arxiv, results_iter):
    mock_client = MagicMock()
    mock_client.results.return_value = results_iter
    mock_arxiv.Client.return_value = mock_client
    mock_arxiv.Search = MagicMock()
    mock_arxiv.SortCriterion = MagicMock()
    mock_arxiv.SortCriterion.Relevance = "relevance"
    mock_arxiv.SortCriterion.SubmittedDate = "date"
    return mock_client


class TestArxivSearch:
    """Test arxiv search with mocked API."""

    async def test_returns_formatted_papers(self):
        mock_results = [
            MockArxivResult(
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer", "Parmar"],
                summary="We propose a new architecture called Transformer.",
                published=MagicMock(strftime=lambda fmt: "2017-06-12"),
                entry_id="http://arxiv.org/abs/1706.03762",
            ),
        ]

        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            _patch_arxiv(mock_arxiv, iter(mock_results))
            from src.tools.arxiv_tool import arxiv_search
            result = await arxiv_search("transformer attention")

            assert "Attention Is All You Need" in result
            assert "Vaswani" in result

    async def test_no_results(self):
        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            _patch_arxiv(mock_arxiv, iter([]))
            from src.tools.arxiv_tool import arxiv_search
            result = await arxiv_search("xyznonexistent123")

            assert "No" in result or "no" in result.lower()

    async def test_author_truncation(self):
        """Papers with 4+ authors should show first 3 + et al."""
        mock_results = [
            MockArxivResult(
                title="Big Paper",
                authors=["Author1", "Author2", "Author3", "Author4", "Author5"],
                summary="A collaborative paper.",
                published=MagicMock(strftime=lambda fmt: "2025-01-01"),
                entry_id="http://arxiv.org/abs/0000.00000",
            ),
        ]

        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            _patch_arxiv(mock_arxiv, iter(mock_results))
            from src.tools.arxiv_tool import arxiv_search
            result = await arxiv_search("big paper")

            assert "et al" in result

    async def test_category_filter_applied(self):
        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            _patch_arxiv(mock_arxiv, iter([]))
            from src.tools.arxiv_tool import arxiv_search
            await arxiv_search("attention", category="cs.AI")

            # Verify the category was prefixed onto the search query
            search_call = mock_arxiv.Search.call_args[1]
            assert "cat:cs.AI" in search_call["query"]

    async def test_typed_kwargs(self):
        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            _patch_arxiv(mock_arxiv, iter([]))
            from src.tools.arxiv_tool import arxiv_search
            result = await arxiv_search("neural networks", max_results=3, sort="date")

            assert len(result) > 0  # Should not crash on typed kwargs
            search_call = mock_arxiv.Search.call_args[1]
            assert search_call["max_results"] == 3


class TestArxivSearchSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        from src.tools.arxiv_tool import ArxivSearchInput
        with pytest.raises(ValidationError):
            ArxivSearchInput()

    def test_invalid_sort_rejected(self):
        from src.tools.arxiv_tool import ArxivSearchInput
        with pytest.raises(ValidationError):
            ArxivSearchInput(query="test", sort="newest")

    def test_max_results_out_of_range_rejected(self):
        from src.tools.arxiv_tool import ArxivSearchInput
        with pytest.raises(ValidationError):
            ArxivSearchInput(query="test", max_results=0)

    def test_valid_input_parses(self):
        from src.tools.arxiv_tool import ArxivSearchInput
        parsed = ArxivSearchInput(query="ML", max_results=10, sort="date", category="cs.LG")
        assert parsed.sort == "date"
        assert parsed.category == "cs.LG"


class TestArxivTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        from src.tools.arxiv_tool import arxiv_tool, ArxivSearchInput
        assert arxiv_tool.name == "arxiv_search"
        assert arxiv_tool.args_schema is ArxivSearchInput

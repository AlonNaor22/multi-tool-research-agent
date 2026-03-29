"""Tests for src/tools/arxiv_tool.py — academic paper search."""

import pytest
from unittest.mock import patch, MagicMock


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
            mock_client = MagicMock()
            mock_client.results.return_value = iter(mock_results)
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"
            mock_arxiv.SortCriterion.SubmittedDate = "date"

            from src.tools.arxiv_tool import search_arxiv
            result = await search_arxiv("transformer attention")

            assert "Attention Is All You Need" in result
            assert "Vaswani" in result

    async def test_no_results(self):
        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            mock_client = MagicMock()
            mock_client.results.return_value = iter([])
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"
            mock_arxiv.SortCriterion.SubmittedDate = "date"

            from src.tools.arxiv_tool import search_arxiv
            result = await search_arxiv("xyznonexistent123")

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
            mock_client = MagicMock()
            mock_client.results.return_value = iter(mock_results)
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"
            mock_arxiv.SortCriterion.SubmittedDate = "date"

            from src.tools.arxiv_tool import search_arxiv
            result = await search_arxiv("big paper")

            assert "et al" in result

    async def test_json_input_with_options(self):
        with patch("src.tools.arxiv_tool.arxiv") as mock_arxiv:
            mock_client = MagicMock()
            mock_client.results.return_value = iter([])
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"
            mock_arxiv.SortCriterion.SubmittedDate = "date"

            from src.tools.arxiv_tool import search_arxiv
            result = await search_arxiv('{"query": "neural networks", "max_results": 3, "sort": "date"}')

            assert len(result) > 0  # Should not crash on JSON input

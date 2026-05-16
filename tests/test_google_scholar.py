"""Tests for src/tools/google_scholar_tool.py -- Semantic Scholar academic search."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError
from tests.conftest import AsyncMockResponse


# Sample Semantic Scholar API response
SEMANTIC_SCHOLAR_RESPONSE = {
    "total": 2,
    "data": [
        {
            "title": "Machine Learning: A Survey",
            "authors": [
                {"name": "John Smith"},
                {"name": "Jane Jones"},
            ],
            "year": 2023,
            "citationCount": 500,
            "abstract": "This paper surveys recent advances in machine learning techniques.",
            "url": "https://www.semanticscholar.org/paper/abc123",
            "externalIds": {"DOI": "10.1234/ml.2023.001"},
            "publicationTypes": ["JournalArticle"],
        },
        {
            "title": "Deep Learning Foundations",
            "authors": [
                {"name": "Alice Brown"},
                {"name": "Bob Davis"},
                {"name": "Carol White"},
                {"name": "Dan Green"},
            ],
            "year": 2022,
            "citationCount": 300,
            "abstract": "A comprehensive introduction to deep learning.",
            "url": "https://www.semanticscholar.org/paper/def456",
            "externalIds": {"ArXiv": "2201.12345"},
            "publicationTypes": ["JournalArticle"],
        },
    ],
}


class TestGoogleScholar:
    """Test academic paper search with mocked HTTP."""

    async def test_returns_formatted_results(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("machine learning")

            assert "Machine Learning: A Survey" in result
            assert "John Smith" in result
            assert "Citations: 500" in result

    async def test_formats_author_et_al(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("deep learning")

            assert "et al." in result

    async def test_doi_url(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("machine learning")

            assert "doi.org" in result

    async def test_arxiv_url_fallback(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("deep learning")

            assert "arxiv.org" in result

    async def test_no_results(self):
        empty_response = {"total": 0, "data": []}
        mock_resp = AsyncMockResponse(json_data=empty_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("xyznonexistent123")

            assert "No academic papers found" in result

    async def test_handles_request_error(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("blocked")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("test")

            assert "Error" in result

    async def test_year_range_kwargs(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            result = await google_scholar("AI", year_from=2020, year_to=2025)

            assert len(result) > 0
            call_kwargs = mock_session.get.call_args
            params = call_kwargs[1].get("params", {}) if call_kwargs[1] else {}
            assert params.get("year") == "2020-2025"

    async def test_year_from_only(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            await google_scholar("neural networks", year_from=2020)

            call_kwargs = mock_session.get.call_args
            params = call_kwargs[1].get("params", {}) if call_kwargs[1] else {}
            assert params.get("year") == "2020-"

    async def test_empty_query(self):
        from src.tools.google_scholar_tool import google_scholar
        result = await google_scholar("")
        assert "Error" in result

    async def test_help_command(self):
        from src.tools.google_scholar_tool import google_scholar
        result = await google_scholar("help")
        assert "Semantic Scholar" in result

    async def test_caching(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import google_scholar, _cache
            _cache.clear()
            await google_scholar("caching test")
            await google_scholar("caching test")  # Should hit cache

            assert mock_session.get.call_count == 1  # Only one actual API call


class TestGoogleScholarSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        from src.tools.google_scholar_tool import GoogleScholarInput
        with pytest.raises(ValidationError):
            GoogleScholarInput()

    def test_year_out_of_range_rejected(self):
        from src.tools.google_scholar_tool import GoogleScholarInput
        with pytest.raises(ValidationError):
            GoogleScholarInput(query="test", year_from=1500)

    def test_valid_input_parses(self):
        from src.tools.google_scholar_tool import GoogleScholarInput
        parsed = GoogleScholarInput(query="AI", year_from=2020, year_to=2025, max_results=5)
        assert parsed.year_from == 2020
        assert parsed.year_to == 2025


class TestGoogleScholarTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        from src.tools.google_scholar_tool import google_scholar_tool, GoogleScholarInput
        assert google_scholar_tool.name == "google_scholar"
        assert google_scholar_tool.args_schema is GoogleScholarInput

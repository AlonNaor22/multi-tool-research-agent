"""Tests for src/tools/google_scholar_tool.py -- Semantic Scholar academic search."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("machine learning")

            assert "Machine Learning: A Survey" in result
            assert "John Smith" in result
            assert "Citations: 500" in result

    async def test_formats_author_et_al(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("deep learning")

            # Second paper has 4 authors, should show 3 + et al.
            assert "et al." in result

    async def test_doi_url(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("machine learning")

            assert "doi.org" in result

    async def test_arxiv_url_fallback(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("deep learning")

            assert "arxiv.org" in result

    async def test_no_results(self):
        empty_response = {"total": 0, "data": []}
        mock_resp = AsyncMockResponse(json_data=empty_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("xyznonexistent123")

            assert "No academic papers found" in result

    async def test_handles_request_error(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("blocked")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search("test")

            assert "Error" in result

    async def test_json_input_with_year_range(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            result = await scholar_search('{"query": "AI", "year_from": 2020, "year_to": 2025}')

            assert len(result) > 0
            # Verify the year param was passed
            call_kwargs = mock_session.get.call_args
            # params are passed as keyword argument
            params = call_kwargs[1].get("params", {}) if call_kwargs[1] else {}
            assert params.get("year") == "2020-2025"

    async def test_empty_query(self):
        from src.tools.google_scholar_tool import scholar_search
        result = await scholar_search("")
        assert "Error" in result

    async def test_help_command(self):
        from src.tools.google_scholar_tool import scholar_search
        result = await scholar_search("help")
        assert "Semantic Scholar" in result

    async def test_year_from_prefix(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            await scholar_search("from 2020: neural networks")

            call_kwargs = mock_session.get.call_args
            params = call_kwargs[1].get("params", {}) if call_kwargs[1] else {}
            assert params.get("year") == "2020-"

    async def test_caching(self):
        mock_resp = AsyncMockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()
            await scholar_search("caching test")
            await scholar_search("caching test")  # Should hit cache

            assert mock_session.get.call_count == 1  # Only one actual API call

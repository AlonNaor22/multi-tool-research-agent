"""Tests for src/tools/google_scholar_tool.py — Semantic Scholar academic search."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


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

    def test_returns_formatted_results(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("machine learning")

            assert "Machine Learning: A Survey" in result
            assert "John Smith" in result
            assert "Citations: 500" in result

    def test_formats_author_et_al(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("deep learning")

            # Second paper has 4 authors, should show 3 + et al.
            assert "et al." in result

    def test_doi_url(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("machine learning")

            assert "doi.org" in result

    def test_arxiv_url_fallback(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("deep learning")

            assert "arxiv.org" in result

    def test_no_results(self):
        empty_response = {"total": 0, "data": []}
        mock_resp = MockResponse(json_data=empty_response, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("xyznonexistent123")

            assert "No academic papers found" in result

    def test_handles_request_error(self):
        import requests
        with patch("src.tools.google_scholar_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("blocked")):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("test")

            assert "Error" in result

    def test_json_input_with_year_range(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search('{"query": "AI", "year_from": 2020, "year_to": 2025}')

            assert len(result) > 0
            # Verify the year param was passed
            call_params = mock_get.call_args[1].get("params", {})
            assert call_params.get("year") == "2020-2025"

    def test_empty_query(self):
        from src.tools.google_scholar_tool import scholar_search
        result = scholar_search("")
        assert "Error" in result

    def test_help_command(self):
        from src.tools.google_scholar_tool import scholar_search
        result = scholar_search("help")
        assert "Semantic Scholar" in result

    def test_year_from_prefix(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.google_scholar_tool import scholar_search
            scholar_search("from 2020: neural networks")

            call_params = mock_get.call_args[1].get("params", {})
            assert call_params.get("year") == "2020-"

    def test_caching(self):
        mock_resp = MockResponse(json_data=SEMANTIC_SCHOLAR_RESPONSE, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.google_scholar_tool import scholar_search, _cache
            _cache.clear()  # Start fresh
            scholar_search("caching test")
            scholar_search("caching test")  # Should hit cache

            assert mock_get.call_count == 1  # Only one actual API call

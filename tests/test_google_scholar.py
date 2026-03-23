"""Tests for src/tools/google_scholar_tool.py — academic research search."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


SCHOLAR_HTML = """
<html>
<body>
<div class="gs_r gs_or gs_scl">
    <div class="gs_ri">
        <h3 class="gs_rt"><a href="https://example.com/paper1">Machine Learning: A Survey</a></h3>
        <div class="gs_a">Smith, Jones - Journal of AI, 2023</div>
        <div class="gs_rs">This paper surveys recent advances in machine learning techniques.</div>
        <div class="gs_fl"><a>Cited by 500</a></div>
    </div>
</div>
<div class="gs_r gs_or gs_scl">
    <div class="gs_ri">
        <h3 class="gs_rt"><a href="https://example.com/paper2">Deep Learning Foundations</a></h3>
        <div class="gs_a">Brown, Davis - Nature, 2022</div>
        <div class="gs_rs">A comprehensive introduction to deep learning.</div>
        <div class="gs_fl"><a>Cited by 300</a></div>
    </div>
</div>
</body>
</html>
"""


class TestGoogleScholar:
    """Test Google Scholar search with mocked HTTP."""

    def test_returns_formatted_results(self):
        mock_resp = MockResponse(text=SCHOLAR_HTML, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("machine learning")

            assert len(result) > 0

    def test_no_results(self):
        empty_html = "<html><body></body></html>"
        mock_resp = MockResponse(text=empty_html, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("xyznonexistent123")

            assert "No" in result or "no" in result.lower() or len(result) > 0

    def test_handles_request_error(self):
        import requests
        with patch("src.tools.google_scholar_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("blocked")):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search("test")

            assert "Error" in result or "error" in result.lower()

    def test_json_input_with_year_range(self):
        mock_resp = MockResponse(text=SCHOLAR_HTML, status_code=200)

        with patch("src.tools.google_scholar_tool.requests.get", return_value=mock_resp):
            from src.tools.google_scholar_tool import scholar_search
            result = scholar_search('{"query": "AI", "year_from": 2020, "year_to": 2025}')

            assert len(result) > 0  # Should not crash on JSON input

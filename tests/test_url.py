"""Tests for src/tools/url_tool.py — web page content extraction."""

import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import MockResponse


SAMPLE_HTML = """
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page for testing">
    <meta name="author" content="Test Author">
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is the main content of the test page.</p>
    <p>It has multiple paragraphs with useful information.</p>
</body>
</html>
"""


class TestUrlFetch:
    """Test URL content fetching with mocked HTTP."""

    def test_extracts_html_content(self):
        mock_resp = MockResponse(
            text=SAMPLE_HTML,
            status_code=200,
            headers={"Content-Type": "text/html"}
        )

        with patch("src.tools.url_tool.requests.get", return_value=mock_resp):
            from src.tools.url_tool import fetch_url_content
            result = fetch_url_content("https://example.com")

            assert "Main Heading" in result or "main content" in result

    def test_extracts_metadata(self):
        mock_resp = MockResponse(
            text=SAMPLE_HTML,
            status_code=200,
            headers={"Content-Type": "text/html"}
        )

        with patch("src.tools.url_tool.requests.get", return_value=mock_resp):
            from src.tools.url_tool import fetch_url_content
            result = fetch_url_content("https://example.com")

            assert "Test Page" in result or "test page" in result.lower()

    def test_handles_404(self):
        from requests.exceptions import HTTPError
        mock_error = HTTPError("404 Not Found")
        mock_error.response = MagicMock(status_code=404)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = mock_error

        with patch("src.tools.url_tool.requests.get", return_value=mock_resp):
            from src.tools.url_tool import fetch_url_content
            result = fetch_url_content("https://example.com/404")

            assert "404" in result or "error" in result.lower()

    def test_handles_timeout(self):
        import requests
        with patch("src.tools.url_tool.requests.get",
                   side_effect=requests.exceptions.Timeout("timeout")):
            from src.tools.url_tool import fetch_url_content
            result = fetch_url_content("https://example.com")

            assert "timed out" in result.lower()

    def test_truncates_long_content(self):
        long_html = f"<html><body>{'A' * 10000}</body></html>"
        mock_resp = MockResponse(
            text=long_html,
            status_code=200,
            headers={"Content-Type": "text/html"}
        )

        with patch("src.tools.url_tool.requests.get", return_value=mock_resp):
            from src.tools.url_tool import fetch_url_content
            result = fetch_url_content("https://example.com")

            # Should be truncated, not the full 10000 chars
            assert len(result) < 10000

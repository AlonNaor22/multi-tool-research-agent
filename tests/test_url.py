"""Tests for src/tools/url_tool.py -- web page content extraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import AsyncMockResponse


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

    async def test_extracts_html_content(self):
        mock_resp = AsyncMockResponse(
            text=SAMPLE_HTML,
            status=200,
            headers={"Content-Type": "text/html"}
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.url_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.url_tool import fetch_url_content
            result = await fetch_url_content("https://example.com")

            assert "Main Heading" in result or "main content" in result

    async def test_extracts_metadata(self):
        mock_resp = AsyncMockResponse(
            text=SAMPLE_HTML,
            status=200,
            headers={"Content-Type": "text/html"}
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.url_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.url_tool import fetch_url_content
            result = await fetch_url_content("https://example.com")

            assert "Test Page" in result or "test page" in result.lower()

    async def test_handles_404(self):
        import aiohttp
        mock_resp = AsyncMockResponse(text="Not Found", status=404)
        mock_resp.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                None, None, status=404, message="Not Found"
            )
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.url_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.url_tool import fetch_url_content
            result = await fetch_url_content("https://example.com/404")

            assert "404" in result or "error" in result.lower()

    async def test_handles_timeout(self):
        import asyncio
        mock_session = MagicMock()
        mock_session.get.side_effect = asyncio.TimeoutError("timeout")

        with patch("src.tools.url_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.url_tool import fetch_url_content
            result = await fetch_url_content("https://example.com")

            assert "timeout" in result.lower()

    async def test_truncates_long_content(self):
        long_html = f"<html><body>{'A' * 10000}</body></html>"
        mock_resp = AsyncMockResponse(
            text=long_html,
            status=200,
            headers={"Content-Type": "text/html"}
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.tools.url_tool.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.url_tool import fetch_url_content
            result = await fetch_url_content("https://example.com")

            # Should be truncated, not the full 10000 chars
            assert len(result) < 10000

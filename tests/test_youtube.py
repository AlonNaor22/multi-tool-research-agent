"""Tests for src/tools/youtube_tool.py — YouTube video search."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


# Minimal YouTube-like HTML with embedded JSON data
YOUTUBE_HTML = """
<html>
<body>
<script>var ytInitialData = {"contents":{"twoColumnSearchResultsRenderer":{"primaryContents":{"sectionListRenderer":{"contents":[{"itemSectionRenderer":{"contents":[{"videoRenderer":{"videoId":"abc123","title":{"runs":[{"text":"Test Video Title"}]},"ownerText":{"runs":[{"text":"Test Channel"}]},"viewCountText":{"simpleText":"1,000 views"},"lengthText":{"simpleText":"10:30"}}}]}}]}}}}};</script>
</body>
</html>
"""

YOUTUBE_HTML_NO_RESULTS = """
<html><body><script>var ytInitialData = {};</script></body></html>
"""


class TestYoutubeSearch:
    """Test YouTube search with mocked HTTP."""

    def test_returns_video_results(self):
        mock_resp = MockResponse(text=YOUTUBE_HTML, status_code=200)

        with patch("src.tools.youtube_tool.requests.get", return_value=mock_resp):
            from src.tools.youtube_tool import youtube_search
            result = youtube_search("test video")

            # Should extract some content from the page
            assert len(result) > 0

    def test_handles_no_results(self):
        mock_resp = MockResponse(text=YOUTUBE_HTML_NO_RESULTS, status_code=200)

        with patch("src.tools.youtube_tool.requests.get", return_value=mock_resp):
            from src.tools.youtube_tool import youtube_search
            result = youtube_search("xyznonexistent123")

            assert len(result) > 0  # Should return message, not crash

    def test_handles_request_error(self):
        import requests
        with patch("src.tools.youtube_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("failed")):
            from src.tools.youtube_tool import youtube_search
            result = youtube_search("test")

            assert "Error" in result or "error" in result.lower()

    def test_empty_query(self):
        from src.tools.youtube_tool import youtube_search
        result = youtube_search("")
        assert len(result) > 0  # Should handle gracefully

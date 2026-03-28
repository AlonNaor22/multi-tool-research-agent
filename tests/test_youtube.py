"""Tests for src/tools/youtube_tool.py — YouTube video search via yt-dlp."""

import pytest
from unittest.mock import patch, MagicMock


# Sample yt-dlp search result
YTDLP_RESULT = {
    "entries": [
        {
            "id": "abc123",
            "title": "Test Video Title",
            "url": "https://www.youtube.com/watch?v=abc123",
            "channel": "Test Channel",
            "uploader": "Test Channel",
            "view_count": 1000,
            "duration": 630,  # 10:30
            "upload_date": "20240115",
            "description": "A test video description for testing.",
        },
        {
            "id": "def456",
            "title": "Second Video",
            "url": "https://www.youtube.com/watch?v=def456",
            "channel": "Another Channel",
            "uploader": "Another Channel",
            "view_count": 50000,
            "duration": 120,
            "upload_date": "20240220",
            "description": "Another test description.",
        },
    ]
}


class TestYoutubeSearch:
    """Test YouTube search with mocked yt-dlp."""

    def test_returns_video_results(self):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = YTDLP_RESULT

        with patch("src.tools.youtube_tool.run_with_timeout", side_effect=lambda f, **kw: f()):
            with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
                from src.tools.youtube_tool import youtube_search, _cache
                _cache.clear()
                result = youtube_search("test video")

                assert "Test Video Title" in result
                assert "Test Channel" in result
                assert "1,000 views" in result

    def test_formats_duration(self):
        from src.tools.youtube_tool import _format_duration
        assert _format_duration(630) == "10:30"
        assert _format_duration(3661) == "1:01:01"
        assert _format_duration(None) == "Unknown"

    def test_formats_views(self):
        from src.tools.youtube_tool import _format_views
        assert _format_views(1000) == "1,000 views"
        assert _format_views(None) == "Unknown views"

    def test_handles_no_results(self):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"entries": []}

        with patch("src.tools.youtube_tool.run_with_timeout", side_effect=lambda f, **kw: f()):
            with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
                from src.tools.youtube_tool import youtube_search, _cache
                _cache.clear()
                result = youtube_search("xyznonexistent123")

                assert "No YouTube videos found" in result

    def test_handles_error(self):
        with patch("src.tools.youtube_tool.run_with_timeout", side_effect=Exception("yt-dlp failed")):
            with patch("yt_dlp.YoutubeDL", side_effect=Exception("yt-dlp failed")):
                from src.tools.youtube_tool import youtube_search, _cache
                _cache.clear()
                result = youtube_search("test")

                assert "Error" in result

    def test_empty_query(self):
        from src.tools.youtube_tool import youtube_search
        result = youtube_search("")
        assert "Error" in result

    def test_result_count_prefix(self):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = YTDLP_RESULT

        with patch("src.tools.youtube_tool.run_with_timeout", side_effect=lambda f, **kw: f()):
            with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
                from src.tools.youtube_tool import youtube_search, _cache
                _cache.clear()
                result = youtube_search("3 results: python tutorial")

                assert len(result) > 0

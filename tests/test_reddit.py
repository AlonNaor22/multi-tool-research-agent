"""Tests for src/tools/reddit_tool.py — Reddit post search."""

import pytest
from unittest.mock import patch
from tests.conftest import MockResponse


REDDIT_API_RESPONSE = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "Best Python libraries for data science",
                    "subreddit_name_prefixed": "r/Python",
                    "score": 1500,
                    "num_comments": 234,
                    "permalink": "/r/Python/comments/abc123/best_python_libraries/",
                    "author": "test_user",
                    "created_utc": 1700000000,
                    "selftext": "I've been exploring different Python libraries for data science...",
                    "url": "https://www.reddit.com/r/Python/comments/abc123/",
                }
            },
            {
                "data": {
                    "title": "Python vs R for machine learning",
                    "subreddit_name_prefixed": "r/datascience",
                    "score": 800,
                    "num_comments": 150,
                    "permalink": "/r/datascience/comments/def456/python_vs_r/",
                    "author": "another_user",
                    "created_utc": 1700100000,
                    "selftext": "Which language is better for ML tasks?",
                    "url": "https://www.reddit.com/r/datascience/comments/def456/",
                }
            },
        ]
    }
}


class TestRedditSearch:
    """Test Reddit search with mocked HTTP."""

    def test_returns_formatted_results(self):
        mock_resp = MockResponse(json_data=REDDIT_API_RESPONSE, status_code=200)

        with patch("src.tools.reddit_tool.requests.get", return_value=mock_resp):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = reddit_search("python data science")

            assert "Best Python libraries" in result
            assert "r/Python" in result
            assert "1.5k" in result  # Score formatting

    def test_no_results(self):
        empty_response = {"data": {"children": []}}
        mock_resp = MockResponse(json_data=empty_response, status_code=200)

        with patch("src.tools.reddit_tool.requests.get", return_value=mock_resp):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = reddit_search("xyznonexistent123")

            assert "No Reddit posts found" in result

    def test_handles_request_error(self):
        import requests
        with patch("src.tools.reddit_tool.requests.get",
                   side_effect=requests.exceptions.ConnectionError("failed")):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = reddit_search("test")

            assert "Error" in result

    def test_subreddit_filter(self):
        mock_resp = MockResponse(json_data=REDDIT_API_RESPONSE, status_code=200)

        with patch("src.tools.reddit_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            reddit_search("r/Python: best libraries")

            call_url = mock_get.call_args[0][0]
            assert "r/Python" in call_url

    def test_empty_query(self):
        from src.tools.reddit_tool import reddit_search
        result = reddit_search("")
        assert "Error" in result

    def test_help_command(self):
        from src.tools.reddit_tool import reddit_search
        result = reddit_search("help")
        assert "FORMAT" in result

    def test_score_formatting(self):
        from src.tools.reddit_tool import _format_score
        assert _format_score(1500) == "1.5k"
        assert _format_score(500) == "500"

    def test_caching(self):
        mock_resp = MockResponse(json_data=REDDIT_API_RESPONSE, status_code=200)

        with patch("src.tools.reddit_tool.requests.get", return_value=mock_resp) as mock_get:
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            reddit_search("cache test query")
            reddit_search("cache test query")  # Should hit cache

            assert mock_get.call_count == 1

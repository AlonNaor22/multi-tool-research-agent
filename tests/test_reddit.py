"""Tests for src/tools/reddit_tool.py -- Reddit post search."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError
from tests.conftest import AsyncMockResponse


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

    async def test_returns_formatted_results(self):
        mock_resp = AsyncMockResponse(json_data=REDDIT_API_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = await reddit_search("python data science")

            assert "Best Python libraries" in result
            assert "r/Python" in result
            assert "1.5k" in result  # Score formatting

    async def test_no_results(self):
        empty_response = {"data": {"children": []}}
        mock_resp = AsyncMockResponse(json_data=empty_response, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = await reddit_search("xyznonexistent123")

            assert "No Reddit posts found" in result

    async def test_handles_request_error(self):
        import aiohttp
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("failed")

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            result = await reddit_search("test")

            assert "Error" in result

    async def test_subreddit_kwarg(self):
        mock_resp = AsyncMockResponse(json_data=REDDIT_API_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            await reddit_search("best libraries", subreddit="Python")

            call_args = mock_session.get.call_args
            call_url = call_args[0][0] if call_args[0] else ""
            assert "r/Python" in call_url

    async def test_sort_and_time_filter_passed(self):
        mock_resp = AsyncMockResponse(json_data=REDDIT_API_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            await reddit_search("ai", sort="top", time_filter="week")

            call_args = mock_session.get.call_args
            params = call_args[1].get("params", {}) if call_args[1] else {}
            assert params.get("sort") == "top"
            assert params.get("t") == "week"

    async def test_empty_query(self):
        from src.tools.reddit_tool import reddit_search
        result = await reddit_search("")
        assert "Error" in result

    async def test_help_command(self):
        from src.tools.reddit_tool import reddit_search
        result = await reddit_search("help")
        assert "PARAMETERS" in result

    def test_score_formatting(self):
        from src.tools.reddit_tool import _format_score
        assert _format_score(1500) == "1.5k"
        assert _format_score(500) == "500"

    async def test_caching(self):
        mock_resp = AsyncMockResponse(json_data=REDDIT_API_RESPONSE, status=200)
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("src.utils.get_aiohttp_session", new_callable=AsyncMock, return_value=mock_session):
            from src.tools.reddit_tool import reddit_search, _cache
            _cache.clear()
            await reddit_search("cache test query")
            await reddit_search("cache test query")  # Should hit cache

            assert mock_session.get.call_count == 1


class TestRedditSearchSchema:
    """Pydantic args_schema validation at the LangChain boundary."""

    def test_missing_query_rejected(self):
        from src.tools.reddit_tool import RedditSearchInput
        with pytest.raises(ValidationError):
            RedditSearchInput()

    def test_invalid_sort_rejected(self):
        from src.tools.reddit_tool import RedditSearchInput
        with pytest.raises(ValidationError):
            RedditSearchInput(query="test", sort="upvotes")

    def test_invalid_time_filter_rejected(self):
        from src.tools.reddit_tool import RedditSearchInput
        with pytest.raises(ValidationError):
            RedditSearchInput(query="test", time_filter="hour")

    def test_valid_input_parses(self):
        from src.tools.reddit_tool import RedditSearchInput
        parsed = RedditSearchInput(
            query="python", subreddit="learnpython",
            sort="top", time_filter="month", max_results=3,
        )
        assert parsed.subreddit == "learnpython"
        assert parsed.sort == "top"
        assert parsed.time_filter == "month"


class TestRedditTool:
    """The BaseTool wrapper exposes the schema to the LangGraph agent."""

    def test_tool_wired_with_schema(self):
        from src.tools.reddit_tool import reddit_tool, RedditSearchInput
        assert reddit_tool.name == "reddit_search"
        assert reddit_tool.args_schema is RedditSearchInput

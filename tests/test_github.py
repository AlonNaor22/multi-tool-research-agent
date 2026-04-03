"""Tests for the GitHub search tool."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def mock_github_response():
    """Sample GitHub API response for repository search."""
    return {
        "total_count": 2,
        "items": [
            {
                "full_name": "langchain-ai/langchain",
                "description": "Build context-aware reasoning applications",
                "stargazers_count": 75000,
                "forks_count": 11000,
                "language": "Python",
                "html_url": "https://github.com/langchain-ai/langchain",
                "updated_at": "2024-03-15T10:00:00Z",
            },
            {
                "full_name": "hwchase17/langchainjs",
                "description": "LangChain for JavaScript",
                "stargazers_count": 8000,
                "forks_count": 1200,
                "language": "TypeScript",
                "html_url": "https://github.com/hwchase17/langchainjs",
                "updated_at": "2024-03-10T08:00:00Z",
            },
        ],
    }


class TestGithubSearch:
    """Tests for the GitHub search tool."""

    @pytest.mark.asyncio
    async def test_repo_search(self, mock_github_response):
        from src.tools.github_tool import github_search

        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_github_response
            result = await github_search("langchain")

        assert "langchain-ai/langchain" in result
        assert "75,000" in result
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_advanced_json_input(self, mock_github_response):
        from src.tools.github_tool import github_search

        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_github_response
            query = json.dumps({"query": "langchain", "type": "repositories", "sort": "stars"})
            result = await github_search(query)

        assert "langchain-ai/langchain" in result
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0] == "/search/repositories"

    @pytest.mark.asyncio
    async def test_code_search(self):
        from src.tools.github_tool import github_search

        code_response = {
            "total_count": 1,
            "items": [
                {
                    "repository": {"full_name": "user/repo"},
                    "path": "src/utils.py",
                    "html_url": "https://github.com/user/repo/blob/main/src/utils.py",
                },
            ],
        }
        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = code_response
            query = json.dumps({"query": "retry decorator", "type": "code"})
            result = await github_search(query)

        assert "user/repo" in result
        assert "src/utils.py" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        from src.tools.github_tool import github_search

        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"total_count": 0, "items": []}
            result = await github_search("xyznonexistent12345")

        assert "No GitHub" in result

    @pytest.mark.asyncio
    async def test_empty_query(self):
        from src.tools.github_tool import github_search
        result = await github_search("")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        from src.tools.github_tool import github_search

        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"error": "GitHub API rate limit exceeded. Try again in a minute."}
            result = await github_search("test")

        assert "rate limit" in result

    @pytest.mark.asyncio
    async def test_issue_search(self):
        from src.tools.github_tool import github_search

        issue_response = {
            "total_count": 1,
            "items": [
                {
                    "title": "Memory leak in agent",
                    "repository_url": "https://api.github.com/repos/user/repo",
                    "state": "open",
                    "html_url": "https://github.com/user/repo/issues/42",
                },
            ],
        }
        with patch("src.tools.github_tool._github_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = issue_response
            query = json.dumps({"query": "memory leak", "type": "issues"})
            result = await github_search(query)

        assert "Memory leak" in result
        assert "open" in result

    def test_tool_wrapper_exists(self):
        from src.tools.github_tool import github_tool
        assert github_tool.name == "github_search"
        assert github_tool.coroutine is not None

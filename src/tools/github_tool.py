"""GitHub search tool via the public REST API."""

import asyncio
from typing import Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_fetch, truncate,
    safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_USER_AGENT, DEFAULT_HTTP_TIMEOUT, DEFAULT_MAX_RESULTS,
    MAX_SEARCH_RESULTS, GITHUB_DESC_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches GitHub via the public REST API for repositories, code,
# issues, or users. Schema is enforced via args_schema.
# ───────────────────────────────────────────────────────────────────

GITHUB_API_BASE = "https://api.github.com"

SearchType = Literal["repositories", "code", "issues", "users"]
SortOrder = Literal["stars", "forks", "updated"]


# Takes (endpoint, params). Calls the GitHub REST API with retry logic.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def _github_api_request(endpoint: str, params: dict) -> dict:
    """Make a request to the GitHub REST API and return the JSON response."""
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"{GITHUB_API_BASE}{endpoint}"
    try:
        return await async_fetch(
            url, params=params, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT,
        )
    except Exception as e:
        if "403" in str(e):
            return {"error": "GitHub API rate limit exceeded. Try again in a minute."}
        raise


# Takes (query, search_type, sort, max_results). Returns formatted results
# for repos, code, issues, or users.
@safe_tool_call("searching GitHub")
async def github_search(
    query: str,
    type: SearchType = "repositories",
    sort: SortOrder = "stars",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Search GitHub with typed parameters and return formatted results."""
    err = require_input(query, "search query")
    if err:
        return err

    max_results = min(int(max_results), MAX_SEARCH_RESULTS)
    search_type = type

    params = {
        "q": query,
        "per_page": max_results,
    }
    if sort and search_type == "repositories":
        params["sort"] = sort
        params["order"] = "desc"

    data = await _github_api_request(f"/search/{search_type}", params)

    if "error" in data:
        return data["error"]

    items = data.get("items", [])
    total_count = data.get("total_count", 0)

    if not items:
        return f"No GitHub {search_type} found for '{query}'"

    formatted = []
    for i, item in enumerate(items, 1):
        if search_type == "repositories":
            name = item.get("full_name", "unknown")
            desc = item.get("description", "No description") or "No description"
            stars = item.get("stargazers_count", 0)
            forks = item.get("forks_count", 0)
            lang = item.get("language", "Unknown")
            url = item.get("html_url", "")
            updated = item.get("updated_at", "")[:10]
            formatted.append(
                f"{i}. **{name}** ({lang})\n"
                f"   {truncate(desc, GITHUB_DESC_MAX_CHARS)}\n"
                f"   Stars: {stars:,} | Forks: {forks:,} | Updated: {updated}\n"
                f"   URL: {url}"
            )
        elif search_type == "code":
            repo = item.get("repository", {}).get("full_name", "unknown")
            path = item.get("path", "unknown")
            url = item.get("html_url", "")
            formatted.append(
                f"{i}. **{repo}** — `{path}`\n"
                f"   URL: {url}"
            )
        elif search_type == "issues":
            title = item.get("title", "No title")
            repo_url = item.get("repository_url", "")
            repo_name = repo_url.split("/repos/")[-1] if "/repos/" in repo_url else ""
            state = item.get("state", "unknown")
            url = item.get("html_url", "")
            formatted.append(
                f"{i}. **{title}** [{state}]\n"
                f"   Repo: {repo_name}\n"
                f"   URL: {url}"
            )
        elif search_type == "users":
            login = item.get("login", "unknown")
            user_type = item.get("type", "User")
            url = item.get("html_url", "")
            formatted.append(
                f"{i}. **{login}** ({user_type})\n"
                f"   URL: {url}"
            )

    header = f"Found {total_count:,} {search_type} for '{query}' (showing top {len(items)}):\n"
    return header + "\n\n".join(formatted)


class GithubSearchInput(BaseModel):
    """Inputs for the github_search tool."""
    query: str = Field(description="GitHub search query string.")
    type: SearchType = Field(
        default="repositories",
        description="What to search: repositories, code, issues, or users.",
    )
    sort: SortOrder = Field(
        default="stars",
        description="Sort order for repository results: stars, forks, or updated.",
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description=f"Number of results to return (1-{MAX_SEARCH_RESULTS}).",
    )


class GithubSearchTool(BaseTool):
    name: str = "github_search"
    description: str = (
        "Search GitHub for repositories, code, issues, or users. Use for finding "
        "open-source projects, code examples, libraries, and developer tools."
        "\n\nUSE FOR:"
        "\n- Finding libraries: query='python web scraping framework'"
        "\n- Code examples: query='async retry decorator', type='code'"
        "\n- Issues/bugs: query='memory leak react', type='issues'"
        "\n- Developers: query='machine learning', type='users'"
    )
    args_schema: Type[BaseModel] = GithubSearchInput

    # Forwards every validated parameter to github_search.
    async def _arun(self, **kwargs) -> str:
        return await github_search(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


github_tool = GithubSearchTool()

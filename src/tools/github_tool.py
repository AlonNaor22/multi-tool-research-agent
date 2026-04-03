"""GitHub search tool for the research agent.

Searches GitHub repositories, code, issues, and users via the public
REST API. No API key required for basic searches (rate-limited to
10 requests/minute for unauthenticated users).

Features:
- Repository search with sorting (stars, forks, updated)
- Code search across public repos
- Issue/PR search
- User/organization lookup
"""

import json
import aiohttp
from langchain_core.tools import Tool
from src.utils import async_retry_on_error, get_aiohttp_session, make_sync
from src.constants import DEFAULT_USER_AGENT, DEFAULT_HTTP_TIMEOUT

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10


@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def _github_api_request(endpoint: str, params: dict) -> dict:
    """Make a request to the GitHub REST API."""
    session = await get_aiohttp_session()
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"{GITHUB_API_BASE}{endpoint}"
    async with session.get(
        url,
        headers=headers,
        params=params,
        timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT),
    ) as resp:
        if resp.status == 403:
            return {"error": "GitHub API rate limit exceeded. Try again in a minute."}
        resp.raise_for_status()
        return await resp.json()


async def github_search(query: str) -> str:
    """Search GitHub for repositories, code, issues, or users.

    Input can be a simple string (searches repos by default) or JSON:
    - Simple: "langchain agent framework"
    - Advanced: {"query": "langchain", "type": "repositories", "sort": "stars", "max_results": 5}

    Supported types: repositories (default), code, issues, users
    Supported sort: stars, forks, updated (repos); indexed (code); created, updated (issues)
    """
    # Parse input
    search_type = "repositories"
    sort = "stars"
    max_results = DEFAULT_MAX_RESULTS

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            search_query = options.get("query", "")
            search_type = options.get("type", "repositories")
            sort = options.get("sort", "stars")
            max_results = min(options.get("max_results", DEFAULT_MAX_RESULTS), MAX_ALLOWED_RESULTS)
        else:
            search_query = query
    except json.JSONDecodeError:
        search_query = query

    if not search_query:
        return "Error: No search query provided."

    try:
        params = {
            "q": search_query,
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
            return f"No GitHub {search_type} found for '{search_query}'"

        # Format results based on type
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
                    f"   {desc[:150]}\n"
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

        header = f"Found {total_count:,} {search_type} for '{search_query}' (showing top {len(items)}):\n"
        return header + "\n\n".join(formatted)

    except Exception as e:
        return f"Error searching GitHub: {str(e)}"


github_tool = Tool(
    name="github_search",
    func=make_sync(github_search),
    coroutine=github_search,
    description=(
        "Search GitHub for repositories, code, issues, or users. Use for finding "
        "open-source projects, code examples, libraries, and developer tools."
        "\n\nUSE FOR:"
        "\n- Finding libraries: 'python web scraping framework'"
        "\n- Code examples: '{\"query\": \"async retry decorator\", \"type\": \"code\"}'"
        "\n- Issues/bugs: '{\"query\": \"memory leak react\", \"type\": \"issues\"}'"
        "\n- Developers: '{\"query\": \"machine learning\", \"type\": \"users\"}'"
        "\n\nSIMPLE: 'search query' (searches repos by stars)"
        "\n\nADVANCED: {\"query\": \"...\", \"type\": \"repositories|code|issues|users\", "
        "\"sort\": \"stars|forks|updated\", \"max_results\": 5}"
    ),
)

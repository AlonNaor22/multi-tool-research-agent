"""Web search tool using DuckDuckGo."""

import asyncio
from typing import Optional, Type

from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    SNIPPET_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Performs general web searches via the DuckDuckGo Search API.
# Schema is enforced by Anthropic's tool-use API via args_schema.
# ───────────────────────────────────────────────────────────────────


# Takes (query, max_results, region). Runs a DuckDuckGo text search in a thread
# with timeout protection. Returns a list of result dicts.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: Optional[str] = None,
):
    """Perform a DuckDuckGo web search asynchronously and return a list of result dicts."""
    ddgs = DDGS()

    search_kwargs = {
        "keywords": query,
        "max_results": max_results,
    }
    if region:
        search_kwargs["region"] = region

    # DDGS has no timeout parameter -- wrap to prevent indefinite blocking
    results = await async_run_with_timeout(
        lambda: list(ddgs.text(**search_kwargs)),
        timeout=DEFAULT_SEARCH_TIMEOUT,
    )

    return results


# Takes (query, max_results, region). Returns numbered results with title,
# URL, and truncated snippet. Errors are caught by safe_tool_call.
@safe_tool_call("performing web search")
async def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: Optional[str] = None,
) -> str:
    """Search the general web via DuckDuckGo and return formatted results."""
    err = require_input(query, "search query")
    if err:
        return err

    max_results = min(int(max_results), MAX_SEARCH_RESULTS)
    results = await async_web_search(query, max_results, region)

    if not results:
        return f"No search results found for '{query}'"

    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("href", result.get("link", "No URL"))
        snippet = result.get("body", result.get("snippet", "No description"))
        snippet = truncate(snippet, SNIPPET_MAX_CHARS)

        formatted_results.append(
            f"{i}. **{title}**\n"
            f"   URL: {url}\n"
            f"   {snippet}"
        )

    header = f"Found {len(results)} results for '{query}':\n"
    return header + "\n\n".join(formatted_results)


class WebSearchInput(BaseModel):
    """Inputs for the web_search tool."""
    query: str = Field(description="Search query string.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description=f"Number of results to return (1-{MAX_SEARCH_RESULTS}).",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region code (e.g. 'us-en', 'uk-en', 'de-de'). Optional.",
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the GENERAL WEB for information from all types of websites. "
        "Returns a mix of blogs, forums, company sites, docs, and articles — "
        "not limited to news."
        "\n\nUSE FOR:"
        "\n- General lookups: 'Tesla stock price today', 'Python 3.12 features'"
        "\n- Comparisons/reviews: 'best laptop 2024', 'React vs Vue'"
        "\n- How-to/tutorials: 'how to deploy Flask on AWS'"
        "\n- Anything not in Wikipedia or needing fresh data from diverse sources"
        "\n\nDO NOT USE FOR:"
        "\n- News articles with sources/dates (use news_search — it returns journalism)"
        "\n- Established facts/history (use wikipedia)"
        "\n- Entity facts like population or GDP (use wikidata)"
        "\n- Scientific constants (use wolfram_alpha)"
        "\n\nRULE: Need GENERAL WEB results? -> web_search. Need NEWS ARTICLES? -> news_search."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    # Forwards every validated parameter to web_search.
    async def _arun(self, **kwargs) -> str:
        return await web_search(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


search_tool = WebSearchTool()

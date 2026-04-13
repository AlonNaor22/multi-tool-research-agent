"""Web search tool using DuckDuckGo."""

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    parse_tool_input, truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    SNIPPET_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Performs general web searches via the DuckDuckGo Search API.
# Formats results with title, URL, and snippet for the agent.
# ───────────────────────────────────────────────────────────────────

# Takes (query, max_results, region). Runs a DuckDuckGo text search in a thread
# with timeout protection. Returns a list of result dicts.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS, region: str = None):
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


# Takes a search query (plain string or JSON with max_results/region).
# Returns numbered results with title, URL, and truncated snippet.
@safe_tool_call("performing web search")
async def web_search(query: str) -> str:
    """Search the GENERAL WEB for information from all types of websites. Returns a mix of blogs, forums, company sites, docs, and articles — not limited to news.

USE FOR:
- General lookups: 'Tesla stock price today', 'Python 3.12 features'
- Comparisons/reviews: 'best laptop 2024', 'React vs Vue'
- How-to/tutorials: 'how to deploy Flask on AWS'
- Anything not in Wikipedia or needing fresh data from diverse sources

DO NOT USE FOR:
- News articles with sources/dates (use news_search — it returns journalism)
- Established facts/history (use wikipedia)
- Entity facts like population or GDP (use wikidata)
- Scientific constants (use wolfram_alpha)

SIMPLE: 'search query' | ADVANCED: {"query": "...", "max_results": 5}

RULE: Need GENERAL WEB results? -> web_search. Need NEWS ARTICLES? -> news_search."""
    # Parse input - could be simple string or JSON with options
    search_query, opts = parse_tool_input(query, {"max_results": DEFAULT_MAX_RESULTS})
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_SEARCH_RESULTS)
    region = opts.get("region")  # e.g., "us-en", "uk-en", "de-de"

    err = require_input(search_query, "search query")
    if err: return err

    results = await async_web_search(search_query, max_results, region)

    if not results:
        return f"No search results found for '{search_query}'"

    # Format results in a structured way
    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("href", result.get("link", "No URL"))
        snippet = result.get("body", result.get("snippet", "No description"))

        # Truncate snippet if too long
        snippet = truncate(snippet, SNIPPET_MAX_CHARS)

        formatted_results.append(
            f"{i}. **{title}**\n"
            f"   URL: {url}\n"
            f"   {snippet}"
        )

    header = f"Found {len(results)} results for '{search_query}':\n"
    return header + "\n\n".join(formatted_results)


search_tool = tool(web_search)

"""Web search tool using DuckDuckGo."""

from duckduckgo_search import DDGS
from src.utils import (
    async_retry_on_error, async_run_with_timeout, create_tool,
    parse_tool_input, truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    SNIPPET_MAX_CHARS,
)


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


@safe_tool_call("performing web search")
async def web_search(query: str) -> str:
    """Takes a search query string, searches DuckDuckGo, returns formatted results."""
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


# Create the LangChain Tool wrapper
search_tool = create_tool(
    "web_search",
    web_search,
    "Search the GENERAL WEB for information from all types of websites. Returns "
    "a mix of blogs, forums, company sites, docs, and articles — not limited to news."
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
    "\n\nSIMPLE: 'search query' | ADVANCED: {\"query\": \"...\", \"max_results\": 5}"
    "\n\nRULE: Need GENERAL WEB results? -> web_search. Need NEWS ARTICLES? -> news_search.",
)

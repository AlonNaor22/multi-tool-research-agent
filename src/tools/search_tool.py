"""Web search tool for the research agent.

Uses DuckDuckGo for free web searches without requiring an API key.
This gives the agent access to current information from the internet.

Features:
- Structured results with title, URL, and snippet
- Configurable result count
- Region filtering support
- Retry logic for reliability
"""

from duckduckgo_search import DDGS
from src.utils import (
    async_retry_on_error, async_run_with_timeout, create_tool,
    parse_tool_input, truncate,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    SNIPPET_MAX_CHARS,
)


@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS, region: str = None):
    """
    Perform a DuckDuckGo web search asynchronously.

    Args:
        query: Search query string
        max_results: Maximum number of results
        region: Optional region filter (e.g., "us-en")

    Returns:
        List of search result dicts.
    """
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


async def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    The query can be a simple search string, or a JSON object with options:
    - Simple: "latest AI news"
    - With options: {"query": "latest AI news", "max_results": 5, "region": "us-en"}

    Args:
        query: Search query string or JSON with options

    Returns:
        Formatted search results with title, URL, and snippet.
    """
    # Parse input - could be simple string or JSON with options
    search_query, opts = parse_tool_input(query, {"max_results": DEFAULT_MAX_RESULTS})
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_SEARCH_RESULTS)
    region = opts.get("region")  # e.g., "us-en", "uk-en", "de-de"

    if not search_query:
        return "Error: No search query provided."

    try:
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

    except Exception as e:
        return f"Error performing web search: {str(e)}"


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

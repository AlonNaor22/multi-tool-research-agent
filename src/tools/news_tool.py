"""News search tool for the research agent.

Uses DuckDuckGo News search to find recent news articles.
No API key required - completely free to use.

Features:
- Configurable time range (day, week, month)
- Configurable result count
- Region filtering
- Structured output with title, source, date, and snippet
"""

from duckduckgo_search import DDGS
from src.utils import (
    async_retry_on_error, async_run_with_timeout, create_tool,
    parse_tool_input, truncate,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    ARTICLE_BODY_MAX_CHARS,
)


# Configuration
DEFAULT_TIMELIMIT = "w"  # Past week


@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_search_news(query: str, max_results: int = DEFAULT_MAX_RESULTS,
                            timelimit: str = DEFAULT_TIMELIMIT, region: str = None):
    """
    Perform a DuckDuckGo news search asynchronously.

    Args:
        query: Search query string
        max_results: Maximum number of results
        timelimit: Time range ('d', 'w', 'm')
        region: Optional region filter

    Returns:
        List of news result dicts.
    """
    ddgs = DDGS()

    search_kwargs = {
        "keywords": query,
        "max_results": max_results,
        "timelimit": timelimit,
    }

    if region:
        search_kwargs["region"] = region

    # DDGS has no timeout parameter -- wrap to prevent indefinite blocking
    results = await async_run_with_timeout(
        lambda: list(ddgs.news(**search_kwargs)),
        timeout=DEFAULT_SEARCH_TIMEOUT,
    )

    return results


async def search_news(query: str) -> str:
    """
    Search for recent news articles using DuckDuckGo News.

    Input can be:
    - Simple topic string: "AI developments"
    - JSON with options: {"query": "AI", "timelimit": "d", "max_results": 10}

    Args:
        query: News topic string or JSON with options

    Returns:
        Formatted news results with titles, sources, and snippets.
    """
    # Parse input
    search_query, opts = parse_tool_input(query, {
        "max_results": DEFAULT_MAX_RESULTS,
        "timelimit": DEFAULT_TIMELIMIT,
    })
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_SEARCH_RESULTS)
    timelimit = opts.get("timelimit", DEFAULT_TIMELIMIT)
    region = opts.get("region")

    if not search_query:
        return "Error: No search query provided."

    # Validate timelimit
    if timelimit not in ("d", "w", "m"):
        timelimit = DEFAULT_TIMELIMIT

    try:
        results = await async_search_news(search_query, max_results, timelimit, region)

        if not results:
            time_desc = {"d": "past day", "w": "past week", "m": "past month"}[timelimit]
            return f"No news found for '{search_query}' in the {time_desc}. Try broader terms or a longer time range."

        # Format the results
        time_desc = {"d": "day", "w": "week", "m": "month"}[timelimit]
        formatted_results = [f"Found {len(results)} news articles for '{search_query}' (past {time_desc}):\n"]

        for i, article in enumerate(results, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            date = article.get('date', 'Unknown date')
            body = article.get('body', 'No description')
            url = article.get('url', '')

            # Truncate body if too long
            body = truncate(body, ARTICLE_BODY_MAX_CHARS)

            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   Source: {source} | Date: {date}\n"
                f"   {body}\n"
                f"   URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error searching news: {str(e)}"


# Create the LangChain Tool wrapper
news_tool = create_tool(
    "news_search",
    search_news,
    "Search NEWS ARTICLES from journalism sources. Returns articles from newspapers, "
    "magazines, and news sites — with publication dates and source names."
    "\n\nUSE FOR:"
    "\n- Breaking news: 'earthquake today', 'election results'"
    "\n- Journalism coverage: 'AI regulation debate', 'climate policy changes'"
    "\n- Time-filtered stories: what happened in the past day/week/month"
    "\n\nDO NOT USE FOR:"
    "\n- General web info (use web_search — broader, not limited to news sources)"
    "\n- Established facts or history (use wikipedia)"
    "\n- Opinions/discussions (use reddit_search)"
    "\n\nSIMPLE: 'artificial intelligence' | ADVANCED: "
    '{\"query\": \"climate\", \"timelimit\": \"d\", \"max_results\": 5}'
    "\n\nTIMELIMIT: 'd' (past day), 'w' (past week, default), 'm' (past month)"
    "\n\nRULE: Need NEWS ARTICLES with sources/dates? -> news_search. "
    "Need general web results? -> web_search.",
)

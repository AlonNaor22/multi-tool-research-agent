"""News search tool using DuckDuckGo News."""

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    parse_tool_input, truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    ARTICLE_BODY_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches recent news articles via the DuckDuckGo News API.
# Supports time-range filtering (day/week/month) and region scoping.
# ───────────────────────────────────────────────────────────────────


# Configuration
DEFAULT_TIMELIMIT = "w"  # Past week


# Takes a query string, result count, time limit, and optional region.
# Returns a list of raw article dicts from DuckDuckGo News.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_search_news(query: str, max_results: int = DEFAULT_MAX_RESULTS,
                            timelimit: str = DEFAULT_TIMELIMIT, region: str = None):
    """Perform a DuckDuckGo news search asynchronously and return a list of result dicts."""
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


# Tool entry point. Parses input options, runs the news search, and formats
# results with title, source, date, body snippet, and URL.
# Returns a formatted multi-article string or an error/empty message.
@safe_tool_call("searching news")
async def news_search(query: str) -> str:
    """Search NEWS ARTICLES from journalism sources. Returns articles from newspapers, magazines, and news sites — with publication dates and source names.

USE FOR:
- Breaking news: 'earthquake today', 'election results'
- Journalism coverage: 'AI regulation debate', 'climate policy changes'
- Time-filtered stories: what happened in the past day/week/month

DO NOT USE FOR:
- General web info (use web_search — broader, not limited to news sources)
- Established facts or history (use wikipedia)
- Opinions/discussions (use reddit_search)

SIMPLE: 'artificial intelligence' | ADVANCED: {"query": "climate", "timelimit": "d", "max_results": 5}

TIMELIMIT: 'd' (past day), 'w' (past week, default), 'm' (past month)

RULE: Need NEWS ARTICLES with sources/dates? -> news_search. Need general web results? -> web_search."""
    # Parse input
    search_query, opts = parse_tool_input(query, {
        "max_results": DEFAULT_MAX_RESULTS,
        "timelimit": DEFAULT_TIMELIMIT,
    })
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_SEARCH_RESULTS)
    timelimit = opts.get("timelimit", DEFAULT_TIMELIMIT)
    region = opts.get("region")

    err = require_input(search_query, "search query")
    if err: return err

    # Validate timelimit
    if timelimit not in ("d", "w", "m"):
        timelimit = DEFAULT_TIMELIMIT

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


news_tool = tool(news_search)

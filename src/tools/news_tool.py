"""News search tool using DuckDuckGo News."""

import asyncio
from typing import Literal, Optional, Type

from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_SEARCH_RESULTS,
    ARTICLE_BODY_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches recent news articles via the DuckDuckGo News API.
# Schema is enforced via args_schema; supports time-range filtering.
# ───────────────────────────────────────────────────────────────────

DEFAULT_TIMELIMIT = "w"  # Past week

TimeLimit = Literal["d", "w", "m"]

TIME_DESC = {"d": "past day", "w": "past week", "m": "past month"}
TIME_SHORT = {"d": "day", "w": "week", "m": "month"}


# Takes (query, max_results, timelimit, region). Returns raw article dicts.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_search_news(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timelimit: str = DEFAULT_TIMELIMIT,
    region: Optional[str] = None,
):
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


# Takes (query, max_results, timelimit, region). Returns formatted article list
# with title, source, date, body snippet, and URL.
@safe_tool_call("searching news")
async def news_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timelimit: TimeLimit = DEFAULT_TIMELIMIT,
    region: Optional[str] = None,
) -> str:
    """Search DuckDuckGo News and return formatted article results."""
    err = require_input(query, "search query")
    if err:
        return err

    max_results = min(int(max_results), MAX_SEARCH_RESULTS)
    if timelimit not in TIME_DESC:
        timelimit = DEFAULT_TIMELIMIT

    results = await async_search_news(query, max_results, timelimit, region)

    if not results:
        return (
            f"No news found for '{query}' in the {TIME_DESC[timelimit]}. "
            "Try broader terms or a longer time range."
        )

    formatted_results = [
        f"Found {len(results)} news articles for '{query}' (past {TIME_SHORT[timelimit]}):\n"
    ]

    for i, article in enumerate(results, 1):
        title = article.get('title', 'No title')
        source = article.get('source', 'Unknown source')
        date = article.get('date', 'Unknown date')
        body = truncate(article.get('body', 'No description'), ARTICLE_BODY_MAX_CHARS)
        url = article.get('url', '')

        formatted_results.append(
            f"{i}. **{title}**\n"
            f"   Source: {source} | Date: {date}\n"
            f"   {body}\n"
            f"   URL: {url}"
        )

    return "\n\n".join(formatted_results)


class NewsSearchInput(BaseModel):
    """Inputs for the news_search tool."""
    query: str = Field(description="News search query string.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description=f"Number of articles to return (1-{MAX_SEARCH_RESULTS}).",
    )
    timelimit: TimeLimit = Field(
        default=DEFAULT_TIMELIMIT,
        description="Time range: 'd' (past day), 'w' (past week), 'm' (past month).",
    )
    region: Optional[str] = Field(
        default=None,
        description="Region code (e.g. 'us-en', 'uk-en'). Optional.",
    )


class NewsSearchTool(BaseTool):
    name: str = "news_search"
    description: str = (
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
        "\n\nTIMELIMIT: 'd' (past day), 'w' (past week, default), 'm' (past month)"
        "\n\nRULE: Need NEWS ARTICLES with sources/dates? -> news_search. Need general web results? -> web_search."
    )
    args_schema: Type[BaseModel] = NewsSearchInput

    # Forwards every validated parameter to news_search.
    async def _arun(self, **kwargs) -> str:
        return await news_search(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


news_tool = NewsSearchTool()

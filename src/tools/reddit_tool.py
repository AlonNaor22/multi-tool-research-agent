"""Reddit search tool using Reddit's public JSON API."""

import asyncio
from typing import List, Dict, Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_fetch,
    truncate, cached_tool, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_USER_AGENT,
    DEFAULT_HTTP_TIMEOUT,
    REDDIT_SEARCH_URL,
    REDDIT_SELFTEXT_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches Reddit posts via the public JSON API. Schema is enforced
# via args_schema (subreddit, sort, time_filter, max_results).
# ───────────────────────────────────────────────────────────────────

SortOrder = Literal["relevance", "top", "new", "comments"]
TimeFilter = Literal["all", "year", "month", "week", "day"]

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS = 10


# Takes (query, max_results, subreddit, sort, time_filter). Calls the Reddit
# JSON API. Returns a list of post dicts.
@cached_tool("reddit")
@async_retry_on_error(max_retries=2, delay=2.0)
async def search_reddit(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    subreddit: Optional[str] = None,
    sort: str = "relevance",
    time_filter: str = "all",
) -> List[Dict]:
    """Search Reddit for posts matching a query and return a list of post dicts."""
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "on", "sort": sort, "t": time_filter, "limit": min(max_results, MAX_RESULTS)}
    else:
        url = REDDIT_SEARCH_URL
        params = {"q": query, "sort": sort, "t": time_filter, "limit": min(max_results, MAX_RESULTS)}

    headers = {"User-Agent": f"{DEFAULT_USER_AGENT} ResearchAgent/1.0"}

    data = await async_fetch(url, params=params, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)

    posts = data.get("data", {}).get("children", [])

    results = []
    for post_wrapper in posts[:max_results]:
        post = post_wrapper.get("data", {})
        if not post:
            continue

        results.append({
            "title": post.get("title", "Untitled"),
            "subreddit": post.get("subreddit_name_prefixed", ""),
            "score": post.get("score", 0),
            "comments": post.get("num_comments", 0),
            "url": f"https://www.reddit.com{post.get('permalink', '')}",
            "author": post.get("author", "[deleted]"),
            "created": post.get("created_utc"),
            "selftext": truncate(post.get("selftext", ""), REDDIT_SELFTEXT_MAX_CHARS),
            "link_url": post.get("url", ""),
        })

    return results


def _format_score(score: int) -> str:
    """Format a Reddit score for display (e.g. 1500 -> '1.5k')."""
    if score >= 1000:
        return f"{score / 1000:.1f}k"
    return str(score)


# Takes a list of post dicts and the original query.
# Returns a formatted multi-line display string with scores and URLs.
def format_results(results: List[Dict], query: str) -> str:
    """Format Reddit search results into a display string."""
    if not results:
        return f"No Reddit posts found for '{query}'. Try different search terms or a specific subreddit."

    lines = [f"Reddit Results for '{query}':", ""]

    for i, post in enumerate(results, 1):
        lines.append(f"{i}. {post['title']}")
        lines.append(f"   {post['subreddit']} | Score: {_format_score(post['score'])} | Comments: {post['comments']}")
        lines.append(f"   Author: u/{post['author']}")
        lines.append(f"   URL: {post['url']}")

        if post.get("selftext"):
            lines.append(f"   Preview: {post['selftext'][:200]}")

        lines.append("")

    return "\n".join(lines)


# Takes (query, max_results, subreddit, sort, time_filter). Returns formatted
# Reddit search results.
@safe_tool_call("searching Reddit")
async def reddit_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    subreddit: Optional[str] = None,
    sort: SortOrder = "relevance",
    time_filter: TimeFilter = "all",
) -> str:
    """Search Reddit with typed parameters and return formatted results."""
    err = require_input(query, "search query")
    if err:
        return err

    if query.strip().lower() in ("help", "?"):
        return _get_help()

    max_results = min(int(max_results), MAX_RESULTS)

    results = await search_reddit(query, max_results, subreddit, sort, time_filter)
    return format_results(results, query)


def _get_help() -> str:
    """Return help text for the Reddit search tool."""
    return """Reddit Search Help:

PARAMETERS:
  query        - Search terms (required)
  max_results  - Number of posts (1-10, default 5)
  subreddit    - Restrict to a specific subreddit (e.g. "Python", "science")
  sort         - "relevance" | "top" | "new" | "comments"
  time_filter  - "all" | "year" | "month" | "week" | "day"

RETURNS:
  - Post title, subreddit, score, comment count
  - Author and post URL
  - Text preview

TIPS:
  - Use subreddit filter for focused discussions (Python, science, askhistorians)
  - Sort by "top" + time_filter="week" for the most upvoted recent posts
  - Reddit is great for opinions, experiences, and recent discussions

EXAMPLES:
  query="best python libraries for data science"
  query="climate change 2024", subreddit="science"
  query="remote work productivity", sort="top", time_filter="month"
"""


# Expose cache for tests (search_reddit._cache)
_cache = search_reddit._cache


class RedditSearchInput(BaseModel):
    """Inputs for the reddit_search tool."""
    query: str = Field(description="Search terms for Reddit posts.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_RESULTS,
        description=f"Number of posts to return (1-{MAX_RESULTS}).",
    )
    subreddit: Optional[str] = Field(
        default=None,
        description="Subreddit name (without 'r/' prefix) to restrict the search.",
    )
    sort: SortOrder = Field(
        default="relevance",
        description="Sort order: relevance, top, new, or comments.",
    )
    time_filter: TimeFilter = Field(
        default="all",
        description="Time filter: all, year, month, week, or day.",
    )


class RedditSearchTool(BaseTool):
    name: str = "reddit_search"
    description: str = (
        "Search Reddit for posts, discussions, and community opinions."
        "\n\nUSE FOR:"
        "\n- Opinions, experiences, recent discussions"
        "\n- Community recommendations"
        "\n- Focused search via subreddit (e.g. 'science', 'askhistorians')"
        "\n- Time-filtered top posts via sort='top' + time_filter='week'"
        "\n\nRETURNS: Post titles, scores, comment counts, subreddit, URLs, text previews."
    )
    args_schema: Type[BaseModel] = RedditSearchInput

    # Forwards every validated parameter to reddit_search.
    async def _arun(self, **kwargs) -> str:
        return await reddit_search(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


reddit_tool = RedditSearchTool()

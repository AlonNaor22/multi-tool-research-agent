"""Reddit search tool using Reddit's public JSON API."""

import re
import json
import asyncio
import aiohttp
from typing import List, Dict

from langchain_core.tools import tool
from src.utils import (
    async_retry_on_error, async_fetch,
    parse_result_count, truncate, cached_tool,
    safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_USER_AGENT,
    DEFAULT_HTTP_TIMEOUT,
    REDDIT_SEARCH_URL,
    REDDIT_SELFTEXT_MAX_CHARS,
)

@cached_tool("reddit")
@async_retry_on_error(max_retries=2, delay=2.0)
async def search_reddit(
    query: str,
    max_results: int = 5,
    subreddit: str = None,
    sort: str = "relevance",
    time_filter: str = "all",
) -> List[Dict]:
    """Search Reddit for posts matching a query and return a list of post dicts."""
    # Build the search URL
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "on", "sort": sort, "t": time_filter, "limit": min(max_results, 10)}
    else:
        url = REDDIT_SEARCH_URL
        params = {"q": query, "sort": sort, "t": time_filter, "limit": min(max_results, 10)}

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


@safe_tool_call("searching Reddit")
async def reddit_search(input_str: str) -> str:
    """Search Reddit for posts, discussions, and community opinions.

FORMAT: 'query', 'r/subreddit: query', 'top week: query'

RETURNS: Post titles, scores, comment counts, subreddit, URLs, text previews.

USE FOR: Finding opinions, experiences, recent discussions, community recommendations.

TIP: Add subreddit filter for focused results (e.g., r/science, r/askhistorians)."""
    input_str = input_str.strip()

    err = require_input(input_str, "search query")
    if err: return err

    if input_str.lower() in ("help", "?"):
        return _get_help()

    max_results = 5
    subreddit = None
    sort = "relevance"
    time_filter = "all"
    query = input_str

    # Try JSON input
    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            query = options.get("query", query)
            max_results = min(options.get("max_results", 5), 10)
            subreddit = options.get("subreddit")
            sort = options.get("sort", "relevance")
            time_filter = options.get("time_filter", "all")
    except json.JSONDecodeError:
        pass

    # Check for "N results:" prefix
    query, max_results = parse_result_count(query, default=max_results)

    # Check for "r/subreddit:" prefix
    sub_match = re.match(r'r/(\w+):\s*(.+)', query, re.IGNORECASE)
    if sub_match:
        subreddit = sub_match.group(1)
        query = sub_match.group(2)

    # Check for sort prefix: "top week:", "new:", etc.
    sort_match = re.match(r'(top|new|comments)\s+(all|year|month|week|day):\s*(.+)', query, re.IGNORECASE)
    if sort_match:
        sort = sort_match.group(1).lower()
        time_filter = sort_match.group(2).lower()
        query = sort_match.group(3)
    elif query.lower().startswith(("top:", "new:")):
        sort = query[:query.index(":")].lower()
        query = query[query.index(":") + 1:].strip()

    results = await search_reddit(query, max_results, subreddit, sort, time_filter)
    return format_results(results, query)


def _get_help() -> str:
    """Return help text for the Reddit search tool."""
    return """Reddit Search Help:

FORMAT:
  python best practices
  r/machinelearning: transformers
  5 results: climate change
  top week: artificial intelligence
  new: latest AI news

OPTIONS:
  N results: query              - Return N results (max 10)
  r/subreddit: query            - Search within a specific subreddit
  top/new/comments [time]: query - Sort by top/new/comments, with time filter

TIME FILTERS: all, year, month, week, day

RETURNS:
  - Post title
  - Subreddit, score, comment count
  - Author and post URL
  - Text preview

TIPS:
  - Use subreddit filter for focused discussions (r/science, r/askhistorians)
  - Sort by "top" to find most upvoted content
  - Reddit is great for opinions, experiences, and recent discussions

EXAMPLES:
  "best python libraries for data science"
  "r/science: climate change 2024"
  "top month: remote work productivity" """


# Expose cache for tests (search_reddit._cache)
_cache = search_reddit._cache

reddit_tool = tool(reddit_search)

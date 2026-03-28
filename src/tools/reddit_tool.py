"""Reddit search tool for the research agent.

Searches Reddit posts and discussions using Reddit's public JSON API.
No authentication required — appends .json to search URLs.
"""

import re
import json
import requests
from typing import List, Dict
from langchain_core.tools import Tool

from src.utils import retry_on_error, TTLCache
from src.constants import (
    DEFAULT_USER_AGENT,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_CACHE_TTL,
    REDDIT_SEARCH_URL,
)

# Cache repeated queries for 5 minutes
_cache = TTLCache(ttl=DEFAULT_CACHE_TTL)


@retry_on_error(max_retries=2, delay=2.0)
def search_reddit(
    query: str,
    max_results: int = 5,
    subreddit: str = None,
    sort: str = "relevance",
    time_filter: str = "all",
) -> List[Dict]:
    """
    Search Reddit for posts matching a query.

    Args:
        query: Search query
        max_results: Maximum results to return (max 10)
        subreddit: Optional subreddit to restrict search to
        sort: Sort order — relevance, new, top, comments
        time_filter: Time filter — all, year, month, week, day

    Returns:
        List of post dictionaries.
    """
    cache_key = _cache.make_key("reddit", query, str(max_results), str(subreddit), sort, time_filter)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Build the search URL
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "on", "sort": sort, "t": time_filter, "limit": min(max_results, 10)}
    else:
        url = REDDIT_SEARCH_URL
        params = {"q": query, "sort": sort, "t": time_filter, "limit": min(max_results, 10)}

    headers = {"User-Agent": f"{DEFAULT_USER_AGENT} ResearchAgent/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    posts = data.get("data", {}).get("children", [])

    results = []
    for post_wrapper in posts[:max_results]:
        post = post_wrapper.get("data", {})
        if not post:
            continue

        # Truncate self-text preview
        selftext = (post.get("selftext") or "")[:300]
        if len(post.get("selftext", "")) > 300:
            selftext += "..."

        results.append({
            "title": post.get("title", "Untitled"),
            "subreddit": post.get("subreddit_name_prefixed", ""),
            "score": post.get("score", 0),
            "comments": post.get("num_comments", 0),
            "url": f"https://www.reddit.com{post.get('permalink', '')}",
            "author": post.get("author", "[deleted]"),
            "created": post.get("created_utc"),
            "selftext": selftext,
            "link_url": post.get("url", ""),
        })

    _cache.set(cache_key, results)
    return results


def _format_score(score: int) -> str:
    """Format a Reddit score for display."""
    if score >= 1000:
        return f"{score / 1000:.1f}k"
    return str(score)


def format_results(results: List[Dict], query: str) -> str:
    """Format Reddit search results for display."""
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


def reddit_search(input_str: str) -> str:
    """
    Search Reddit for posts and discussions.

    Supports formats:
    - "python best practices"
    - "r/machinelearning: transformers"
    - "5 results: climate change"
    - "top week: artificial intelligence"

    Args:
        input_str: Search query with optional filters

    Returns:
        Formatted search results
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty search query"

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
    count_match = re.match(r'(\d+)\s+results?:\s*(.+)', query, re.IGNORECASE)
    if count_match:
        max_results = min(int(count_match.group(1)), 10)
        query = count_match.group(2)

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

    try:
        results = search_reddit(query, max_results, subreddit, sort, time_filter)
        return format_results(results, query)
    except requests.exceptions.RequestException as e:
        return f"Error searching Reddit: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _get_help() -> str:
    """Return help text."""
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


# Create the LangChain Tool wrapper
reddit_tool = Tool(
    name="reddit_search",
    func=reddit_search,
    description=(
        "Search Reddit for posts, discussions, and community opinions. "
        "\n\nFORMAT: 'query', 'r/subreddit: query', 'top week: query'"
        "\n\nRETURNS: Post titles, scores, comment counts, subreddit, URLs, text previews."
        "\n\nUSE FOR: Finding opinions, experiences, recent discussions, community recommendations."
        "\n\nTIP: Add subreddit filter for focused results (e.g., r/science, r/askhistorians)."
    )
)

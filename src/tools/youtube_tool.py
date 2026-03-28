"""YouTube search tool using yt-dlp for reliable video discovery.

Replaces the previous fragile web-scraping approach with yt-dlp's
ytsearch protocol, which is actively maintained and robust against
YouTube frontend changes.
"""

import re
import json
from typing import List, Dict
from langchain_core.tools import Tool

from src.utils import retry_on_error, run_with_timeout, TTLCache
from src.constants import DEFAULT_SEARCH_TIMEOUT, DEFAULT_CACHE_TTL

# Cache repeated queries for 5 minutes
_cache = TTLCache(ttl=DEFAULT_CACHE_TTL)


def _format_duration(seconds) -> str:
    """Convert seconds to H:MM:SS or MM:SS string."""
    if not seconds:
        return "Unknown"
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "Unknown"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_views(count) -> str:
    """Format a view count with commas, e.g. 1,234,567 views."""
    if count is None:
        return "Unknown views"
    try:
        return f"{int(count):,} views"
    except (TypeError, ValueError):
        return "Unknown views"


@retry_on_error(max_retries=2, delay=1.0)
def search_youtube_ytdlp(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search YouTube via yt-dlp's ytsearch protocol.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of video dictionaries.
    """
    cache_key = _cache.make_key("youtube", query, str(max_results))
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "playlist_items": f"1:{max_results}",
    }

    search_url = f"ytsearch{max_results}:{query}"

    def _do_search():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(search_url, download=False)

    data = run_with_timeout(_do_search, timeout=DEFAULT_SEARCH_TIMEOUT)

    results = []
    entries = data.get("entries") or []

    for entry in entries[:max_results]:
        if not entry:
            continue

        video_id = entry.get("id", "")
        results.append({
            "title": entry.get("title", "Unknown"),
            "video_id": video_id,
            "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            "channel": entry.get("channel") or entry.get("uploader") or "Unknown",
            "views": _format_views(entry.get("view_count")),
            "duration": _format_duration(entry.get("duration")),
            "published": entry.get("upload_date") or "",
            "description": (entry.get("description") or "")[:200],
        })

    _cache.set(cache_key, results)
    return results


def format_results(results: List[Dict], query: str) -> str:
    """Format YouTube video search results into a readable multi-line string."""
    if not results:
        return f"No YouTube videos found for '{query}'"

    lines = [f"YouTube Search Results for '{query}':", ""]

    for i, video in enumerate(results, 1):
        lines.append(f"{i}. {video['title']}")
        lines.append(f"   Channel: {video['channel']}")
        lines.append(f"   Duration: {video['duration']} | Views: {video['views']}")
        if video.get("published"):
            lines.append(f"   Published: {video['published']}")
        lines.append(f"   URL: {video['url']}")
        if video.get("description"):
            lines.append(f"   Description: {video['description'][:150]}...")
        lines.append("")

    return "\n".join(lines)


def youtube_search(input_str: str) -> str:
    """
    Search YouTube for videos.

    Supports formats:
    - "python tutorial"
    - "5 results: quantum computing"

    Args:
        input_str: Search query, optionally with result count

    Returns:
        Formatted search results
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty search query"

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Parse for custom result count
    max_results = 5
    query = input_str

    # Check for "N results:" prefix
    count_match = re.match(r'(\d+)\s+results?:\s*(.+)', input_str, re.IGNORECASE)
    if count_match:
        max_results = min(int(count_match.group(1)), 10)
        query = count_match.group(2)

    # Check for "search:" prefix
    if query.lower().startswith("search:"):
        query = query[7:].strip()

    try:
        results = search_youtube_ytdlp(query, max_results)
        return format_results(results, query)
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"


def _get_help() -> str:
    """Return help text."""
    return """YouTube Search Help:

FORMAT:
  python tutorial
  machine learning explained
  5 results: quantum computing

OPTIONS:
  N results: query  - Return N results (max 10)
  search: query     - Explicit search prefix

RETURNS:
  - Video title
  - Channel name
  - Duration and view count
  - Publication date
  - Video URL
  - Description snippet

TIPS:
  - Use specific terms for better results
  - Add "tutorial", "explained", or "how to" for educational content
  - Add year for recent content (e.g., "python 2024")"""


# Create the LangChain Tool wrapper
youtube_tool = Tool(
    name="youtube_search",
    func=youtube_search,
    description=(
        "Search YouTube for videos on any topic. "
        "\n\nFORMAT: 'python tutorial', '5 results: machine learning'"
        "\n\nRETURNS: Video titles, channels, duration, views, URLs, and descriptions."
        "\n\nUSE FOR: Finding tutorials, explanations, lectures, demonstrations, news coverage."
    )
)

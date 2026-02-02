"""YouTube search tool for the research agent.

Searches YouTube for videos on a topic and returns video information.

Uses the yt-dlp library for searching (no API key required).
Falls back to web scraping if needed.
"""

import re
import json
import requests
from typing import List, Dict, Optional
from urllib.parse import quote_plus
from langchain_core.tools import Tool

from src.utils import retry_on_error


@retry_on_error(max_retries=2, delay=1.0)
def search_youtube_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search YouTube using web scraping approach.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of video dictionaries with title, url, channel, etc.
    """
    # Use YouTube's search URL
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(search_url, headers=headers, timeout=15)
    response.raise_for_status()

    html = response.text

    # Extract video data from the page's initial data
    # YouTube embeds JSON data in the page
    results = []

    # Find video IDs and titles using regex patterns
    # Pattern for video renderer data
    video_pattern = r'"videoId":"([^"]+)".*?"title":\{"runs":\[\{"text":"([^"]+)"\}\]'
    channel_pattern = r'"ownerText":\{"runs":\[\{"text":"([^"]+)"'
    view_pattern = r'"viewCountText":\{"simpleText":"([^"]+)"'
    length_pattern = r'"lengthText":\{"accessibility":\{"accessibilityData":\{"label":"([^"]+)"\}\},"simpleText":"([^"]+)"'

    # Try to find the ytInitialData JSON
    data_match = re.search(r'var ytInitialData = ({.*?});', html)

    if data_match:
        try:
            data = json.loads(data_match.group(1))

            # Navigate to video results
            contents = (
                data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
            )

            for section in contents:
                items = (
                    section.get("itemSectionRenderer", {})
                    .get("contents", [])
                )

                for item in items:
                    video_renderer = item.get("videoRenderer")
                    if not video_renderer:
                        continue

                    video_id = video_renderer.get("videoId", "")
                    if not video_id:
                        continue

                    # Extract title
                    title_runs = video_renderer.get("title", {}).get("runs", [])
                    title = title_runs[0].get("text", "Unknown") if title_runs else "Unknown"

                    # Extract channel
                    channel_runs = video_renderer.get("ownerText", {}).get("runs", [])
                    channel = channel_runs[0].get("text", "Unknown") if channel_runs else "Unknown"

                    # Extract view count
                    view_count = video_renderer.get("viewCountText", {}).get("simpleText", "Unknown views")

                    # Extract duration
                    duration = video_renderer.get("lengthText", {}).get("simpleText", "Unknown")

                    # Extract description snippet
                    desc_snippets = video_renderer.get("detailedMetadataSnippets", [])
                    description = ""
                    if desc_snippets:
                        snippet_runs = desc_snippets[0].get("snippetText", {}).get("runs", [])
                        description = "".join(run.get("text", "") for run in snippet_runs)

                    # Extract publish date
                    published = video_renderer.get("publishedTimeText", {}).get("simpleText", "")

                    results.append({
                        "title": title,
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "channel": channel,
                        "views": view_count,
                        "duration": duration,
                        "published": published,
                        "description": description[:200] if description else "",
                    })

                    if len(results) >= max_results:
                        break

                if len(results) >= max_results:
                    break

        except json.JSONDecodeError:
            pass

    # Fallback: simple regex extraction if JSON parsing failed
    if not results:
        # Find video IDs
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        titles = re.findall(r'"title":\{"runs":\[\{"text":"([^"]+)"\}\]', html)

        seen_ids = set()
        for i, vid_id in enumerate(video_ids):
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)

            title = titles[i] if i < len(titles) else "Unknown Title"

            results.append({
                "title": title,
                "video_id": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "channel": "Unknown",
                "views": "Unknown",
                "duration": "Unknown",
                "published": "",
                "description": "",
            })

            if len(results) >= max_results:
                break

    return results


def format_results(results: List[Dict], query: str) -> str:
    """Format search results for display."""
    if not results:
        return f"No YouTube videos found for '{query}'"

    lines = [f"YouTube Search Results for '{query}':", ""]

    for i, video in enumerate(results, 1):
        lines.append(f"{i}. {video['title']}")
        lines.append(f"   Channel: {video['channel']}")
        lines.append(f"   Duration: {video['duration']} | Views: {video['views']}")
        if video.get('published'):
            lines.append(f"   Published: {video['published']}")
        lines.append(f"   URL: {video['url']}")
        if video.get('description'):
            lines.append(f"   Description: {video['description'][:150]}...")
        lines.append("")

    return "\n".join(lines)


def youtube_search(input_str: str) -> str:
    """
    Search YouTube for videos.

    Supports formats:
    - "python tutorial"
    - "search: machine learning explained"
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
        max_results = min(int(count_match.group(1)), 10)  # Cap at 10
        query = count_match.group(2)

    # Check for "search:" prefix
    if query.lower().startswith("search:"):
        query = query[7:].strip()

    try:
        results = search_youtube_web(query, max_results)
        return format_results(results, query)
    except requests.exceptions.RequestException as e:
        return f"Error searching YouTube: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


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

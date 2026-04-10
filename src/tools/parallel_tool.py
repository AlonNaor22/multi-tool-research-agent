"""Parallel execution tool for the research agent.

This is a "meta-tool" that runs multiple searches concurrently using
asyncio.gather(). Instead of the agent making 3 sequential searches,
it can use this tool to run all 3 at once, significantly speeding up research.

Features:
- Supports web, wikipedia, news, and arxiv searches
- Runs searches concurrently using asyncio.gather
- Smart result truncation based on content type
- Maximum 10 parallel searches
"""

import json
import asyncio
from typing import Dict, List
from src.constants import TRUNCATION_PRESERVE_RATIO
from src.utils import create_tool

# Import the async search functions from our tools
from src.tools.search_tool import web_search
from src.tools.wikipedia_tool import search_wikipedia
from src.tools.news_tool import search_news
from src.tools.arxiv_tool import search_arxiv


# Timeout for the entire parallel operation (seconds)
PARALLEL_TIMEOUT = 60

# Result truncation limits by type
TRUNCATION_LIMITS = {
    "web": 600,
    "wikipedia": 800,
    "news": 700,
    "arxiv": 800,
    "default": 500,
}


def get_search_function(search_type: str):
    """Get the appropriate async search function based on type."""
    search_functions = {
        "web": web_search,
        "wikipedia": search_wikipedia,
        "news": search_news,
        "arxiv": search_arxiv,
    }
    return search_functions.get(search_type.lower())


def truncate_result(result: str, search_type: str) -> str:
    """Intelligently truncate result based on type."""
    limit = TRUNCATION_LIMITS.get(search_type, TRUNCATION_LIMITS["default"])

    if len(result) <= limit:
        return result

    # Try to truncate at a sentence boundary
    truncated = result[:limit]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')

    # Use the later of period or newline as cut point
    cut_point = max(last_period, last_newline)

    # Only use the sentence/line boundary if it preserves at least 70% of
    # the target length; otherwise a hard cut loses less useful content.
    if cut_point > limit * TRUNCATION_PRESERVE_RATIO:
        truncated = result[:cut_point + 1]
    else:
        truncated = result[:limit]

    return truncated + "..."


async def execute_single_search(search_spec: Dict) -> Dict:
    """
    Execute a single search asynchronously and return the result with metadata.

    Args:
        search_spec: Dict with 'type' and 'query' keys

    Returns:
        Dict with 'type', 'query', 'result', and 'success' keys
    """
    search_type = search_spec.get("type", "web")
    query = search_spec.get("query", "")

    search_func = get_search_function(search_type)

    if search_func is None:
        return {
            "type": search_type,
            "query": query,
            "result": f"Unknown search type: {search_type}. Use 'web', 'wikipedia', 'news', or 'arxiv'.",
            "success": False
        }

    try:
        result = await search_func(query)
        return {
            "type": search_type,
            "query": query,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "type": search_type,
            "query": query,
            "result": f"Error: {str(e)}",
            "success": False
        }


async def parallel_search(input_str: str) -> str:
    """
    Execute multiple searches concurrently using asyncio.gather.

    INPUT FORMAT:
    {
        "searches": [
            {"type": "web", "query": "Tesla stock price"},
            {"type": "wikipedia", "query": "Tesla company"},
            {"type": "news", "query": "Tesla"},
            {"type": "arxiv", "query": "electric vehicles battery"}
        ]
    }

    Args:
        input_str: JSON string with "searches" array

    Returns:
        Formatted string with all search results
    """
    # Parse input
    try:
        spec = json.loads(input_str)
    except json.JSONDecodeError as e:
        return (
            f"Error: Invalid JSON. {e}\n\n"
            "Expected format:\n"
            '{\"searches\": [{\"type\": \"web\", \"query\": \"...\"}, ...]}'
        )

    searches = spec.get("searches", [])

    if not searches:
        return "Error: No searches provided. Include a 'searches' array."

    if len(searches) > 10:
        return "Error: Maximum 10 parallel searches allowed."

    # Validate each search
    for i, s in enumerate(searches):
        if "query" not in s:
            return f"Error: Search {i+1} is missing 'query' field."

    # Execute all searches concurrently with asyncio.gather
    try:
        tasks = [execute_single_search(search) for search in searches]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=PARALLEL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"Error: Parallel search timed out after {PARALLEL_TIMEOUT}s"

    # Process results (gather with return_exceptions may return Exception objects)
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({
                "type": searches[i].get("type", "unknown"),
                "query": searches[i].get("query", ""),
                "result": f"Execution error: {str(result)}",
                "success": False
            })
        else:
            processed.append(result)

    # Count successes
    success_count = sum(1 for r in processed if r["success"])

    # Format the results
    output_lines = [
        f"Parallel search completed: {success_count}/{len(processed)} successful\n"
    ]

    for i, r in enumerate(processed, 1):
        status = "SUCCESS" if r["success"] else "FAILED"
        search_type = r["type"].upper()

        # Truncate result smartly
        truncated_result = truncate_result(r["result"], r["type"])

        output_lines.append(
            f"--- [{search_type}] {status} ---\n"
            f"Query: {r['query']}\n"
            f"{truncated_result}\n"
        )

    return "\n".join(output_lines)


# Create the LangChain Tool wrapper
parallel_tool = create_tool(
    name="parallel_search",
    async_fn=parallel_search,
    description=(
        "Execute multiple searches in parallel for faster results. "
        "Use this when you need to gather information from multiple sources at once. "
        "\n\nSUPPORTED TYPES: web, wikipedia, news, arxiv"
        "\n\nFORMAT:"
        '\n{"searches": [{"type": "web", "query": "..."}, {"type": "arxiv", "query": "..."}]}'
        "\n\nEXAMPLE:"
        '\n{"searches": ['
        '{"type": "web", "query": "Tesla stock 2024"}, '
        '{"type": "wikipedia", "query": "Tesla Inc"}, '
        '{"type": "news", "query": "Tesla"}, '
        '{"type": "arxiv", "query": "electric vehicle battery"}'
        ']}'
        "\n\nLIMITS: Maximum 10 searches per call. All run simultaneously."
    ),
)

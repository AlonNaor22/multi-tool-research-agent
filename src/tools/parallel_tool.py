"""Parallel search meta-tool — runs multiple searches concurrently via asyncio.gather."""

import json
import asyncio
from typing import Dict
from src.constants import TRUNCATION_PRESERVE_RATIO
from langchain_core.tools import tool

# Import the async search functions from our tools
from src.tools.search_tool import web_search
from src.tools.wikipedia_tool import wikipedia
from src.tools.news_tool import news_search
from src.tools.arxiv_tool import arxiv_search

# ─── Module overview ───────────────────────────────────────────────
# Meta-tool that dispatches multiple search queries (web, wikipedia,
# news, arxiv) concurrently via asyncio.gather and merges results.
# ───────────────────────────────────────────────────────────────────


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


# Takes a search type string (web/wikipedia/news/arxiv).
# Returns the matching async search function, or None if unknown.
def get_search_function(search_type: str):
    """Return the async search function for a given type, or None."""
    search_functions = {
        "web": web_search,
        "wikipedia": wikipedia,
        "news": news_search,
        "arxiv": arxiv_search,
    }
    return search_functions.get(search_type.lower())


# Takes a result string and its search type. Truncates to a type-specific
# character limit, preferring sentence/line boundaries.
def truncate_result(result: str, search_type: str) -> str:
    """Truncate a search result to a type-specific character limit."""
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


# Takes a search spec dict with "type" and "query" keys.
# Returns a dict with type, query, result string, and success boolean.
async def execute_single_search(search_spec: Dict) -> Dict:
    """Run one search and return a dict with type, query, result, and success."""
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


# Tool entry point. Takes a JSON string with a "searches" array.
# Runs all searches concurrently, truncates each result, and returns
# a combined summary with per-source status.
async def parallel_search(input_str: str) -> str:
    """Execute multiple searches in parallel for faster results. Use this when you need to gather information from multiple sources at once.

    SUPPORTED TYPES: web, wikipedia, news, arxiv

    FORMAT:
    {"searches": [{"type": "web", "query": "..."}, {"type": "arxiv", "query": "..."}]}

    EXAMPLE:
    {"searches": [{"type": "web", "query": "Tesla stock 2024"}, {"type": "wikipedia", "query": "Tesla Inc"}, {"type": "news", "query": "Tesla"}, {"type": "arxiv", "query": "electric vehicle battery"}]}

    LIMITS: Maximum 10 searches per call. All run simultaneously."""
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


parallel_tool = tool(parallel_search)

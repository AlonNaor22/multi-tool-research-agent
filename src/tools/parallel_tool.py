"""Parallel execution tool for the research agent.

This is a "meta-tool" that runs multiple searches in parallel.
Instead of the agent making 3 sequential searches, it can use this
tool to run all 3 at once, significantly speeding up research.

KEY CONCEPT: ThreadPoolExecutor
-------------------------------
Python's concurrent.futures module provides ThreadPoolExecutor,
which manages a pool of worker threads. When we submit multiple
tasks, they run simultaneously (truly parallel for I/O operations
like HTTP requests).

WHY THIS WORKS:
--------------
Our tools mostly do I/O (network requests to APIs). Python's GIL
(Global Interpreter Lock) allows threads to run in parallel during
I/O operations. So web searches, API calls, etc. benefit greatly
from threading.

USAGE:
------
The agent provides a JSON input with multiple queries:
{
    "searches": [
        {"type": "web", "query": "Tesla stock price"},
        {"type": "wikipedia", "query": "Tesla company"},
        {"type": "web", "query": "SpaceX latest launch"}
    ]
}

All searches run in parallel and results are returned together.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from langchain_core.tools import Tool

# Import the actual search functions from our tools
from src.tools.search_tool import web_search
from src.tools.wikipedia_tool import search_wikipedia
from src.tools.news_tool import search_news


# Maximum number of parallel workers
MAX_WORKERS = 5

# Timeout for each individual search (seconds)
SEARCH_TIMEOUT = 30


def get_search_function(search_type: str):
    """
    Get the appropriate search function based on type.

    This is a simple dispatcher that maps type names to functions.
    """
    search_functions = {
        "web": web_search,
        "wikipedia": search_wikipedia,
        "news": search_news,
    }
    return search_functions.get(search_type.lower())


def execute_single_search(search_spec: Dict) -> Dict:
    """
    Execute a single search and return the result with metadata.

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
            "result": f"Unknown search type: {search_type}. Use 'web', 'wikipedia', or 'news'.",
            "success": False
        }

    try:
        result = search_func(query)
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


def parallel_search(input_str: str) -> str:
    """
    Execute multiple searches in parallel.

    HOW IT WORKS:
    1. Parse the JSON input to get list of searches
    2. Create a ThreadPoolExecutor with MAX_WORKERS threads
    3. Submit all searches to the executor
    4. Wait for all to complete and collect results
    5. Format and return combined results

    The key insight is that while one thread waits for a network
    response, other threads can be making their own requests.
    This is much faster than sequential execution.

    Args:
        input_str: JSON string with "searches" array

    Returns:
        Formatted string with all search results
    """
    # Parse input
    try:
        spec = json.loads(input_str)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON. {e}\n\nExpected format:\n{{\"searches\": [{{\"type\": \"web\", \"query\": \"...\"}}, ...]}}"

    searches = spec.get("searches", [])

    if not searches:
        return "Error: No searches provided. Include a 'searches' array."

    if len(searches) > 10:
        return "Error: Maximum 10 parallel searches allowed."

    # Validate each search
    for i, s in enumerate(searches):
        if "query" not in s:
            return f"Error: Search {i+1} is missing 'query' field."

    # Execute searches in parallel
    results = []

    # ThreadPoolExecutor creates a pool of worker threads
    # max_workers controls how many run simultaneously
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all searches - this returns immediately
        # Each submit() schedules the function to run in a thread
        future_to_search = {
            executor.submit(execute_single_search, search): search
            for search in searches
        }

        # as_completed() yields futures as they finish
        # This is more efficient than waiting for all in order
        for future in as_completed(future_to_search, timeout=SEARCH_TIMEOUT):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle any unexpected errors
                search = future_to_search[future]
                results.append({
                    "type": search.get("type", "unknown"),
                    "query": search.get("query", ""),
                    "result": f"Execution error: {str(e)}",
                    "success": False
                })

    # Format the results
    output_lines = [f"Parallel search completed ({len(results)} searches):\n"]

    for i, r in enumerate(results, 1):
        status = "✓" if r["success"] else "✗"
        output_lines.append(f"--- Result {i} [{r['type'].upper()}] {status} ---")
        output_lines.append(f"Query: {r['query']}")
        output_lines.append(f"Result: {r['result'][:500]}...")  # Truncate long results
        output_lines.append("")

    return "\n".join(output_lines)


# Create the LangChain Tool wrapper
parallel_tool = Tool(
    name="parallel_search",
    func=parallel_search,
    description=(
        "Execute multiple searches in parallel for faster results. "
        "Use this when you need to gather information from multiple sources simultaneously. "
        "Input must be a JSON string with a 'searches' array. Each search needs 'type' "
        "('web', 'wikipedia', or 'news') and 'query' fields. "
        "Example: "
        '{\"searches\": ['
        '{\"type\": \"web\", \"query\": \"Tesla stock price 2024\"}, '
        '{\"type\": \"wikipedia\", \"query\": \"Tesla Inc\"}, '
        '{\"type\": \"news\", \"query\": \"Tesla\"}]} '
        "Maximum 10 searches per call. All searches run at the same time, "
        "making this much faster than searching one by one."
    )
)

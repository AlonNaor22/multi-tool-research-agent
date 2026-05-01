"""Parallel search meta-tool — runs multiple searches concurrently via asyncio.gather."""

import asyncio
from typing import Dict, List, Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.constants import TRUNCATION_PRESERVE_RATIO
from src.tools.search_tool import web_search
from src.tools.wikipedia_tool import wikipedia
from src.tools.news_tool import news_search
from src.tools.arxiv_tool import arxiv_search

# ─── Module overview ───────────────────────────────────────────────
# Meta-tool that dispatches multiple search queries (web, wikipedia,
# news, arxiv) concurrently via asyncio.gather and merges results.
# Schema is enforced by Anthropic's tool-use API via args_schema.
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

    truncated = result[:limit]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')

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


# Takes a list of search specs. Dispatches all concurrently with a wall-clock
# timeout, formats per-source status, and returns a combined summary string.
async def parallel_search(searches: List[Dict]) -> str:
    """Dispatch a list of search specs concurrently; return a formatted summary."""
    try:
        tasks = [execute_single_search(s) for s in searches]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=PARALLEL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"Error: Parallel search timed out after {PARALLEL_TIMEOUT}s"

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({
                "type": searches[i].get("type", "unknown"),
                "query": searches[i].get("query", ""),
                "result": f"Execution error: {str(result)}",
                "success": False,
            })
        else:
            processed.append(result)

    success_count = sum(1 for r in processed if r["success"])

    output_lines = [
        f"Parallel search completed: {success_count}/{len(processed)} successful\n"
    ]

    for r in processed:
        status = "SUCCESS" if r["success"] else "FAILED"
        search_type = r["type"].upper()
        truncated_result = truncate_result(r["result"], r["type"])

        output_lines.append(
            f"--- [{search_type}] {status} ---\n"
            f"Query: {r['query']}\n"
            f"{truncated_result}\n"
        )

    return "\n".join(output_lines)


class SearchSpec(BaseModel):
    """A single parallel search specification."""
    type: Literal["web", "wikipedia", "news", "arxiv"] = Field(
        default="web",
        description="Search backend to use.",
    )
    query: str = Field(description="Search query string.")


class ParallelSearchInput(BaseModel):
    """Inputs for the parallel_search tool."""
    searches: List[SearchSpec] = Field(
        min_length=1,
        max_length=10,
        description="1 to 10 searches to dispatch concurrently.",
    )


class ParallelSearchTool(BaseTool):
    name: str = "parallel_search"
    description: str = (
        "Execute multiple searches in parallel for faster results. "
        "Use when you need to gather information from multiple sources at once."
        "\n\nSUPPORTED TYPES: web, wikipedia, news, arxiv"
        "\n\nEXAMPLE: searches=[{\"type\":\"web\",\"query\":\"Tesla 2024\"}, "
        "{\"type\":\"wikipedia\",\"query\":\"Tesla Inc\"}]"
        "\n\nLIMITS: 1–10 searches per call. All run simultaneously."
    )
    args_schema: Type[BaseModel] = ParallelSearchInput

    # LangChain validates kwargs through args_schema and passes SearchSpec
    # instances. Normalize to dicts so parallel_search stays Pydantic-agnostic.
    async def _arun(self, searches: List[SearchSpec]) -> str:
        return await parallel_search([s.model_dump() for s in searches])

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


parallel_tool = ParallelSearchTool()

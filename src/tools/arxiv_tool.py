"""ArXiv academic paper search tool."""

import asyncio
from typing import Literal, Optional, Type

import arxiv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    truncate, safe_tool_call, require_input,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_ARXIV_RESULTS,
    ABSTRACT_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches the ArXiv pre-print repository for academic papers in STEM.
# Schema is enforced via args_schema (category, sort, max_results).
# ───────────────────────────────────────────────────────────────────

SortOrder = Literal["relevance", "date"]

# Common ArXiv categories
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.SE": "Software Engineering",
    "cs.CR": "Cryptography and Security",
    "stat.ML": "Machine Learning (Statistics)",
    "math.OC": "Optimization and Control",
    "physics": "Physics (all)",
    "quant-ph": "Quantum Physics",
    "math": "Mathematics (all)",
    "econ": "Economics",
}


# Takes (search_query, max_results, sort_by). Queries the ArXiv API with retry.
@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_search_arxiv(
    search_query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    sort_by: str = "relevance",
):
    """Perform an ArXiv search asynchronously and return a list of paper objects."""
    client = arxiv.Client()

    if sort_by == "date":
        sort_criterion = arxiv.SortCriterion.SubmittedDate
    else:
        sort_criterion = arxiv.SortCriterion.Relevance

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    # Execute the search with a timeout -- the arxiv library has no
    # built-in timeout, so a slow network could block indefinitely.
    papers = await async_run_with_timeout(
        lambda: list(client.results(search)),
        timeout=DEFAULT_SEARCH_TIMEOUT,
    )

    return papers


# Takes (query, max_results, sort, category, full_abstract). Searches ArXiv.
# Returns formatted paper list with titles, authors, dates, categories, abstracts.
@safe_tool_call("searching ArXiv")
async def arxiv_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    sort: SortOrder = "relevance",
    category: Optional[str] = None,
    full_abstract: bool = False,
) -> str:
    """Search ArXiv for pre-prints and return a formatted paper list."""
    err = require_input(query, "search query")
    if err:
        return err

    max_results = min(int(max_results), MAX_ARXIV_RESULTS)

    search_query = f"cat:{category} AND {query}" if category else query

    papers = await async_search_arxiv(search_query, max_results, sort)

    if not papers:
        return f"No academic papers found for '{query}'"

    sort_desc = "newest first" if sort == "date" else "most relevant"
    results = [f"Found {len(papers)} papers on ArXiv ({sort_desc}):\n"]

    for i, paper in enumerate(papers, 1):
        # Get first 3 authors (some papers have dozens)
        authors = ", ".join([author.name for author in paper.authors[:3]])
        if len(paper.authors) > 3:
            authors += f" et al. ({len(paper.authors)} total)"

        summary = paper.summary.replace('\n', ' ')
        if not full_abstract:
            summary = truncate(summary, ABSTRACT_MAX_CHARS)

        categories = ", ".join(paper.categories[:3])

        results.append(
            f"{i}. **{paper.title}**\n"
            f"   Authors: {authors}\n"
            f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"   Categories: {categories}\n"
            f"   URL: {paper.entry_id}\n"
            f"   Abstract: {summary}\n"
        )

    return "\n".join(results)


class ArxivSearchInput(BaseModel):
    """Inputs for the arxiv_search tool."""
    query: str = Field(description="Search query for ArXiv papers.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_ARXIV_RESULTS,
        description=f"Number of papers to return (1-{MAX_ARXIV_RESULTS}).",
    )
    sort: SortOrder = Field(
        default="relevance",
        description="Sort order: 'relevance' (most relevant) or 'date' (newest first).",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "ArXiv category filter (e.g. 'cs.AI', 'cs.LG', 'physics', 'stat.ML'). "
            "Optional — narrows results to one category."
        ),
    )
    full_abstract: bool = Field(
        default=False,
        description="If true, return full abstracts; otherwise truncate to a snippet.",
    )


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Search ArXiv for PRE-PRINTS — the latest unpublished research in STEM fields. "
        "ArXiv papers are first-to-publish but NOT peer-reviewed."
        "\n\nUSE FOR:"
        "\n- Cutting-edge research: newest papers in AI, ML, physics, math, CS, statistics"
        "\n- Free full-text PDFs of recent papers"
        "\n- Filtering by arXiv category (cs.AI, cs.LG, physics, math, stat.ML, etc.)"
        "\n\nDO NOT USE FOR:"
        "\n- Peer-reviewed/published papers (use google_scholar — it covers journals)"
        "\n- Non-STEM fields: medicine, history, humanities, social science (use google_scholar)"
        "\n\nRULE: Need the LATEST pre-prints in STEM? -> arxiv. Need PUBLISHED, peer-reviewed papers? -> google_scholar."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    # Forwards every validated parameter to arxiv_search.
    async def _arun(self, **kwargs) -> str:
        return await arxiv_search(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


arxiv_tool = ArxivSearchTool()

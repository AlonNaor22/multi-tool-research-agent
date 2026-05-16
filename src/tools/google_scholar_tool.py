"""Academic paper search tool using the Semantic Scholar API."""

import asyncio
from typing import List, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_retry_on_error, async_fetch, cached_tool,
    truncate, safe_tool_call, require_input,
)
from src.constants import (
    SEMANTIC_SCHOLAR_BASE_URL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MAX_RESULTS,
    MAX_SEARCH_RESULTS,
    ABSTRACT_MAX_CHARS,
    SNIPPET_MAX_CHARS,
)

# ─── Module overview ───────────────────────────────────────────────
# Searches published academic papers via the Semantic Scholar API.
# Schema is enforced via args_schema (year_from/year_to, max_results).
# ───────────────────────────────────────────────────────────────────

# Fields to request from the Semantic Scholar API
_PAPER_FIELDS = "title,authors,year,citationCount,abstract,url,externalIds,publicationTypes"


# Takes (query, max_results, year_from, year_to). Queries Semantic Scholar.
# Returns a list of paper dicts with title, authors, year, citations, abstract, url.
@cached_tool("scholar")
@async_retry_on_error(max_retries=2, delay=2.0)
async def search_semantic_scholar(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Dict]:
    """Search Semantic Scholar and return a list of paper dictionaries."""
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"

    params = {
        "query": query,
        "limit": min(max_results, MAX_SEARCH_RESULTS),
        "fields": _PAPER_FIELDS,
    }

    # Year range filter (Semantic Scholar uses "year" param as "YYYY-YYYY")
    if year_from and year_to:
        params["year"] = f"{year_from}-{year_to}"
    elif year_from:
        params["year"] = f"{year_from}-"
    elif year_to:
        params["year"] = f"-{year_to}"

    data = await async_fetch(url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)

    papers = data.get("data", [])

    results = []
    for paper in papers:
        # Build a URL — prefer DOI, then Semantic Scholar page
        paper_url = ""
        external_ids = paper.get("externalIds") or {}
        if external_ids.get("DOI"):
            paper_url = f"https://doi.org/{external_ids['DOI']}"
        elif external_ids.get("ArXiv"):
            paper_url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
        elif paper.get("url"):
            paper_url = paper["url"]

        # Format authors (first 3 + et al.)
        authors_list = paper.get("authors") or []
        author_names = [a.get("name", "") for a in authors_list[:3]]
        if len(authors_list) > 3:
            author_names.append("et al.")
        authors_str = ", ".join(author_names) if author_names else "Unknown"

        results.append({
            "title": paper.get("title", "Untitled"),
            "authors": authors_str,
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
            "abstract": truncate(paper.get("abstract") or "", ABSTRACT_MAX_CHARS),
            "url": paper_url,
        })

    return results


# Takes (results, query). Formats paper dicts into a numbered display string.
def format_results(results: List[Dict], query: str) -> str:
    """Format a list of paper dicts into a display string."""
    if not results:
        return f"No academic papers found for '{query}'. Try different search terms or broader keywords."

    lines = [f"Academic Paper Results for '{query}':", ""]

    for i, paper in enumerate(results, 1):
        lines.append(f"{i}. {paper['title']}")
        lines.append(f"   Authors: {paper['authors']}")

        if paper.get("year"):
            lines.append(f"   Year: {paper['year']}")

        if paper.get("citations"):
            lines.append(f"   Citations: {paper['citations']}")

        if paper.get("url"):
            lines.append(f"   URL: {paper['url']}")

        if paper.get("abstract"):
            lines.append(f"   Abstract: {truncate(paper['abstract'], SNIPPET_MAX_CHARS)}")

        lines.append("")

    lines.append("TIP: Use pdf_reader tool to read full papers if URL points to a PDF.")

    return "\n".join(lines)


# Takes (query, max_results, year_from, year_to). Searches Semantic Scholar
# and returns formatted academic paper results.
@safe_tool_call("searching academic papers")
async def google_scholar(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> str:
    """Search Semantic Scholar with typed parameters and return formatted results."""
    err = require_input(query, "search query")
    if err:
        return err

    if query.strip().lower() in ("help", "?"):
        return _get_help()

    max_results = min(int(max_results), MAX_SEARCH_RESULTS)

    results = await search_semantic_scholar(query, max_results, year_from, year_to)
    return format_results(results, query)


# Returns help text listing supported options and examples.
def _get_help() -> str:
    """Return help text for the scholar search tool."""
    return """Academic Paper Search Help (powered by Semantic Scholar):

PARAMETERS:
  query        - Search terms (required)
  max_results  - Number of papers to return (1-10, default 5)
  year_from    - Earliest publication year
  year_to      - Latest publication year

RETURNS:
  - Paper title, authors, publication year
  - Citation count
  - URL (DOI or ArXiv link)
  - Abstract snippet

TIPS:
  - Covers ALL academic fields (STEM, humanities, medicine, social sciences)
  - Use specific academic terms for better results
  - Add "review" for overview papers
  - Add "meta-analysis" for comprehensive studies
  - Combine with pdf_reader to read full papers

EXAMPLES:
  query="paleoclimate levant bronze age"
  query="deep learning survey", year_from=2015
  query="roman climate reconstruction", year_from=2000, year_to=2010"""


# Expose cache for test clearing
_cache = search_semantic_scholar._cache


class GoogleScholarInput(BaseModel):
    """Inputs for the google_scholar tool."""
    query: str = Field(description="Search terms for academic papers.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description=f"Number of papers to return (1-{MAX_SEARCH_RESULTS}).",
    )
    year_from: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2100,
        description="Earliest publication year (inclusive). Optional.",
    )
    year_to: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2100,
        description="Latest publication year (inclusive). Optional.",
    )


class GoogleScholarTool(BaseTool):
    name: str = "google_scholar"
    description: str = (
        "Search PUBLISHED, peer-reviewed academic papers across ALL fields via Semantic Scholar. "
        "Covers journals, conferences, and theses — with citation counts."
        "\n\nUSE FOR:"
        "\n- Any academic field: medicine, history, social science, law, STEM, humanities"
        "\n- Papers with citation counts (to judge impact)"
        "\n- Year-filtered searches (year_from, year_to)"
        "\n- Published, vetted research (not pre-prints)"
        "\n\nDO NOT USE FOR:"
        "\n- Latest unpublished pre-prints (use arxiv_search — it has newest STEM papers)"
        "\n- General web info (use web_search)"
        "\n\nRULE: Need PUBLISHED papers with citations? -> google_scholar. Need the NEWEST pre-prints in STEM? -> arxiv."
    )
    args_schema: Type[BaseModel] = GoogleScholarInput

    # Forwards every validated parameter to google_scholar.
    async def _arun(self, **kwargs) -> str:
        return await google_scholar(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


google_scholar_tool = GoogleScholarTool()

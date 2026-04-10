"""Academic paper search tool using the Semantic Scholar API."""

import re
import asyncio
import aiohttp
from typing import List, Dict, Optional

from langchain_core.tools import tool
from src.utils import (
    async_retry_on_error, async_fetch, cached_tool,
    parse_tool_input, parse_result_count, truncate,
    safe_tool_call, require_input,
)
from src.constants import (
    SEMANTIC_SCHOLAR_BASE_URL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MAX_RESULTS,
    MAX_SEARCH_RESULTS,
    ABSTRACT_MAX_CHARS,
    SNIPPET_MAX_CHARS,
)

# Fields to request from the Semantic Scholar API
_PAPER_FIELDS = "title,authors,year,citationCount,abstract,url,externalIds,publicationTypes"


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


@safe_tool_call("searching academic papers")
async def google_scholar(input_str: str) -> str:
    """Search PUBLISHED, peer-reviewed academic papers across ALL fields via Semantic Scholar. Covers journals, conferences, and theses — with citation counts.

USE FOR:
- Any academic field: medicine, history, social science, law, STEM, humanities
- Papers with citation counts (to judge impact)
- Year-filtered searches: 'from 2020: topic' or '2010-2020: topic'
- Published, vetted research (not pre-prints)

DO NOT USE FOR:
- Latest unpublished pre-prints (use arxiv_search — it has newest STEM papers)
- General web info (use web_search)

FORMAT: 'roman empire climate', 'from 2020: topic', '2010-2020: paleoclimate levant'

RULE: Need PUBLISHED papers with citations? -> google_scholar. Need the NEWEST pre-prints in STEM? -> arxiv."""
    input_str = input_str.strip()

    err = require_input(input_str, "search query")
    if err: return err

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Parse options
    query, opts = parse_tool_input(input_str, {
        "max_results": DEFAULT_MAX_RESULTS,
    })
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_SEARCH_RESULTS)
    year_from = opts.get("year_from")
    year_to = opts.get("year_to")

    # Check for "N results:" prefix
    query, max_results = parse_result_count(query, max_results, MAX_SEARCH_RESULTS)

    # Check for "from YEAR:" prefix
    from_match = re.match(r'from\s+(\d{4}):\s*(.+)', query, re.IGNORECASE)
    if from_match:
        year_from = int(from_match.group(1))
        query = from_match.group(2)

    # Check for "YEAR-YEAR:" range prefix
    range_match = re.match(r'(\d{4})\s*-\s*(\d{4}):\s*(.+)', query, re.IGNORECASE)
    if range_match:
        year_from = int(range_match.group(1))
        year_to = int(range_match.group(2))
        query = range_match.group(3)

    # Check for "until YEAR:" or "to YEAR:" prefix
    to_match = re.match(r'(?:until|to)\s+(\d{4}):\s*(.+)', query, re.IGNORECASE)
    if to_match:
        year_to = int(to_match.group(1))
        query = to_match.group(2)

    results = await search_semantic_scholar(query, max_results, year_from, year_to)
    return format_results(results, query)


def _get_help() -> str:
    """Return help text for the scholar search tool."""
    return """Academic Paper Search Help (powered by Semantic Scholar):

FORMAT:
  paleoclimate ancient israel
  5 results: machine learning transformers
  from 2020: climate change effects
  2010-2020: roman empire climate
  until 2000: ancient history archaeology

OPTIONS:
  N results: query     - Return N results (max 10)
  from YEAR: query     - Papers from YEAR onwards
  until YEAR: query    - Papers up to YEAR
  YEAR-YEAR: query     - Papers within year range

RETURNS:
  - Paper title
  - Authors
  - Publication year
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
  "paleoclimate levant bronze age"
  "from 2015: deep learning survey"
  "2000-2010: roman climate reconstruction" """


# Expose cache for test clearing
_cache = search_semantic_scholar._cache

google_scholar_tool = tool(google_scholar)

"""Academic paper search tool using the Semantic Scholar API.

Searches for published research across ALL academic fields — STEM,
humanities, medicine, social sciences, etc. Returns structured data
(title, authors, year, citations, abstract, URL) without web scraping.

Free API, no key required.
"""

import re
import json
import requests
from typing import List, Dict, Optional
from langchain_core.tools import Tool

from src.utils import retry_on_error, TTLCache
from src.constants import (
    SEMANTIC_SCHOLAR_BASE_URL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_CACHE_TTL,
)

# Cache repeated queries for 5 minutes
_cache = TTLCache(ttl=DEFAULT_CACHE_TTL)

# Fields to request from the Semantic Scholar API
_PAPER_FIELDS = "title,authors,year,citationCount,abstract,url,externalIds,publicationTypes"


@retry_on_error(max_retries=2, delay=2.0)
def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Dict]:
    """
    Search Semantic Scholar for academic papers.

    Args:
        query: Search query
        max_results: Maximum number of results (max 10)
        year_from: Filter results from this year onwards
        year_to: Filter results up to this year

    Returns:
        List of paper dictionaries.
    """
    cache_key = _cache.make_key("scholar", query, str(max_results), str(year_from), str(year_to))
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"

    params = {
        "query": query,
        "limit": min(max_results, 10),
        "fields": _PAPER_FIELDS,
    }

    # Year range filter (Semantic Scholar uses "year" param as "YYYY-YYYY")
    if year_from and year_to:
        params["year"] = f"{year_from}-{year_to}"
    elif year_from:
        params["year"] = f"{year_from}-"
    elif year_to:
        params["year"] = f"-{year_to}"

    response = requests.get(url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    response.raise_for_status()

    data = response.json()
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
            "abstract": (paper.get("abstract") or "")[:400],
            "url": paper_url,
        })

    _cache.set(cache_key, results)
    return results


def format_results(results: List[Dict], query: str) -> str:
    """Format search results for display."""
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
            lines.append(f"   Abstract: {paper['abstract'][:250]}...")

        lines.append("")

    lines.append("TIP: Use pdf_reader tool to read full papers if URL points to a PDF.")

    return "\n".join(lines)


def scholar_search(input_str: str) -> str:
    """
    Search for academic papers across all fields.

    Supports formats:
    - "climate change effects"
    - "5 results: machine learning"
    - "from 2020: neural networks"
    - "2010-2020: paleoclimate israel"

    Args:
        input_str: Search query with optional filters

    Returns:
        Formatted search results
    """
    input_str = input_str.strip()

    if not input_str:
        return "Error: Empty search query"

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Parse options
    max_results = 5
    year_from = None
    year_to = None
    query = input_str

    # Try JSON input first
    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            query = options.get("query", query)
            max_results = min(options.get("max_results", 5), 10)
            year_from = options.get("year_from")
            year_to = options.get("year_to")
    except json.JSONDecodeError:
        pass

    # Check for "N results:" prefix
    count_match = re.match(r'(\d+)\s+results?:\s*(.+)', query, re.IGNORECASE)
    if count_match:
        max_results = min(int(count_match.group(1)), 10)
        query = count_match.group(2)

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

    try:
        results = search_semantic_scholar(query, max_results, year_from, year_to)
        return format_results(results, query)
    except requests.exceptions.RequestException as e:
        return f"Error searching academic papers: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _get_help() -> str:
    """Return help text."""
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


# Create the LangChain Tool wrapper
google_scholar_tool = Tool(
    name="google_scholar",
    func=scholar_search,
    description=(
        "Search for published academic research across ALL fields using Semantic Scholar. "
        "\n\nBEST FOR: History, humanities, medicine, social sciences, archaeology, climate history, ancient studies."
        "\n\nNOT FOR: Cutting-edge AI/ML or CS research (use arxiv_search instead)."
        "\n\nFORMAT: 'roman empire climate', 'from 2020: topic', '2010-2020: paleoclimate levant'"
        "\n\nFEATURES: Year filters, citation counts, covers peer-reviewed journals and theses."
        "\n\nRETURNS: Titles, authors, year, citation count, URLs, abstracts."
    )
)

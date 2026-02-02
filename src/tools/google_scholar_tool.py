"""Google Scholar search tool for the research agent.

Searches Google Scholar for academic papers, citations, and research.

Useful for historical research, scientific studies, and academic topics.
Uses web scraping (no API key required).
"""

import re
import requests
from typing import List, Dict
from urllib.parse import quote_plus
from langchain_core.tools import Tool

from src.utils import retry_on_error


@retry_on_error(max_retries=2, delay=2.0)
def search_google_scholar(query: str, max_results: int = 5, year_from: int = None, year_to: int = None) -> List[Dict]:
    """
    Search Google Scholar for academic papers.

    Args:
        query: Search query
        max_results: Maximum number of results (max 10)
        year_from: Filter results from this year onwards
        year_to: Filter results up to this year

    Returns:
        List of paper dictionaries with title, authors, snippet, url, etc.
    """
    # Build the search URL
    base_url = "https://scholar.google.com/scholar"

    params = {
        "q": query,
        "hl": "en",
        "num": min(max_results, 10),
    }

    # Add year filters if specified
    if year_from:
        params["as_ylo"] = year_from
    if year_to:
        params["as_yhi"] = year_to

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    response = requests.get(base_url, params=params, headers=headers, timeout=15)
    response.raise_for_status()

    html = response.text
    results = []

    # Parse results using regex (Google Scholar's structure)
    # Each result is in a div with class "gs_r gs_or gs_scl"

    # Pattern for title and link
    title_pattern = r'<h3 class="gs_rt"[^>]*>(?:<span[^>]*>[^<]*</span>)?(?:<a[^>]*href="([^"]*)"[^>]*>)?(.+?)</(?:a>)?</h3>'

    # Pattern for authors/source line
    author_pattern = r'<div class="gs_a">(.+?)</div>'

    # Pattern for snippet
    snippet_pattern = r'<div class="gs_rs">(.+?)</div>'

    # Pattern for citation count
    cite_pattern = r'Cited by (\d+)'

    # Find all result blocks
    result_blocks = re.findall(r'<div class="gs_r gs_or gs_scl"[^>]*>(.+?)</div>\s*</div>\s*</div>', html, re.DOTALL)

    # If that doesn't work, try alternative pattern
    if not result_blocks:
        result_blocks = re.findall(r'<div class="gs_ri">(.+?)</div>\s*(?=<div class="gs_ri">|<div class="gs_r"|$)', html, re.DOTALL)

    for block in result_blocks[:max_results]:
        paper = {}

        # Extract title and URL
        title_match = re.search(title_pattern, block, re.DOTALL)
        if title_match:
            paper["url"] = title_match.group(1) if title_match.group(1) else ""
            # Clean HTML tags from title
            title = title_match.group(2)
            title = re.sub(r'<[^>]+>', '', title)
            paper["title"] = title.strip()
        else:
            # Try simpler pattern
            simple_title = re.search(r'<a[^>]*>([^<]+)</a>', block)
            if simple_title:
                paper["title"] = simple_title.group(1).strip()
            else:
                continue  # Skip if no title found

        # Extract authors and source
        author_match = re.search(author_pattern, block, re.DOTALL)
        if author_match:
            author_text = author_match.group(1)
            # Clean HTML tags
            author_text = re.sub(r'<[^>]+>', '', author_text)
            paper["authors_source"] = author_text.strip()

            # Try to extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', author_text)
            if year_match:
                paper["year"] = year_match.group(0)

        # Extract snippet
        snippet_match = re.search(snippet_pattern, block, re.DOTALL)
        if snippet_match:
            snippet = snippet_match.group(1)
            # Clean HTML tags
            snippet = re.sub(r'<[^>]+>', '', snippet)
            paper["snippet"] = snippet.strip()[:300]

        # Extract citation count
        cite_match = re.search(cite_pattern, block)
        if cite_match:
            paper["citations"] = cite_match.group(1)

        if paper.get("title"):
            results.append(paper)

    # Fallback: simpler extraction if regex didn't work well
    if not results:
        # Try to find any titles
        all_titles = re.findall(r'<h3 class="gs_rt"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]+)', html, re.DOTALL)
        for url, title in all_titles[:max_results]:
            results.append({
                "title": title.strip(),
                "url": url,
                "authors_source": "See link for details",
                "snippet": ""
            })

    return results


def format_results(results: List[Dict], query: str) -> str:
    """Format search results for display."""
    if not results:
        return f"No Google Scholar results found for '{query}'. Try different search terms or check if the topic has academic coverage."

    lines = [f"Google Scholar Results for '{query}':", ""]

    for i, paper in enumerate(results, 1):
        lines.append(f"{i}. {paper.get('title', 'Unknown Title')}")

        if paper.get('authors_source'):
            lines.append(f"   Source: {paper['authors_source']}")

        if paper.get('year'):
            lines.append(f"   Year: {paper['year']}")

        if paper.get('citations'):
            lines.append(f"   Citations: {paper['citations']}")

        if paper.get('url'):
            lines.append(f"   URL: {paper['url']}")

        if paper.get('snippet'):
            lines.append(f"   Summary: {paper['snippet'][:200]}...")

        lines.append("")

    lines.append("TIP: Use pdf_reader tool to read full papers if URL points to a PDF.")

    return "\n".join(lines)


def scholar_search(input_str: str) -> str:
    """
    Search Google Scholar for academic papers.

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

    # Check for "N results:" prefix
    count_match = re.match(r'(\d+)\s+results?:\s*(.+)', input_str, re.IGNORECASE)
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
        results = search_google_scholar(query, max_results, year_from, year_to)
        return format_results(results, query)
    except requests.exceptions.RequestException as e:
        return f"Error searching Google Scholar: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _get_help() -> str:
    """Return help text."""
    return """Google Scholar Search Help:

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
  - Authors and source/journal
  - Publication year
  - Citation count
  - URL to paper
  - Abstract snippet

TIPS:
  - Use specific academic terms for better results
  - Add "review" for overview papers
  - Add "meta-analysis" for comprehensive studies
  - Use year filters for historical research
  - Combine with pdf_reader to read full papers

EXAMPLES:
  "paleoclimate levant bronze age"
  "from 2015: deep learning survey"
  "2000-2010: roman climate reconstruction"
  "holocene climate mediterranean review" """


# Create the LangChain Tool wrapper
google_scholar_tool = Tool(
    name="google_scholar",
    func=scholar_search,
    description=(
        "Search Google Scholar for academic papers, research, and scientific studies. "
        "\n\nFORMAT: 'climate change effects', 'from 2020: neural networks', '2010-2020: paleoclimate'"
        "\n\nRETURNS: Paper titles, authors, year, citations, URLs, and abstracts."
        "\n\nUSE FOR: Academic research, historical studies, scientific topics, literature reviews."
        "\n\nBETTER THAN web_search for: Historical questions, scientific data, peer-reviewed sources."
    )
)

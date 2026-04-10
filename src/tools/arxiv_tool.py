"""ArXiv academic paper search tool for the research agent.

ArXiv (arxiv.org) is a free repository of academic papers in physics,
mathematics, computer science, and more. This tool allows the agent
to search for academic papers and retrieve their metadata.

Features:
- Configurable result count
- Sorting by relevance or date
- Category filtering (cs.AI, physics, math, etc.)
- Full abstract option

No API key required - ArXiv is completely free and open.
"""

import arxiv
from src.utils import (
    async_retry_on_error, async_run_with_timeout, create_tool,
    parse_tool_input, truncate,
)
from src.constants import (
    DEFAULT_SEARCH_TIMEOUT, DEFAULT_MAX_RESULTS, MAX_ARXIV_RESULTS,
    ABSTRACT_MAX_CHARS,
)

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


@async_retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
async def async_search_arxiv(search_query: str, max_results: int = DEFAULT_MAX_RESULTS,
                             sort_by: str = "relevance"):
    """
    Perform an ArXiv search asynchronously.

    Args:
        search_query: Search query string (may include category prefix)
        max_results: Maximum number of results
        sort_by: Sort criterion ('relevance' or 'date')

    Returns:
        List of paper objects.
    """
    client = arxiv.Client()

    # Determine sort criterion
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


async def search_arxiv(query: str) -> str:
    """
    Search ArXiv for academic papers.

    Input can be:
    - Simple query string: "transformer neural networks"
    - JSON with options: {"query": "transformers", "max_results": 10, "sort": "date"}

    Args:
        query: Search query string or JSON with options

    Returns:
        Formatted string with paper titles, authors, and summaries.
    """
    # Parse input
    search_query, opts = parse_tool_input(query, {
        "max_results": DEFAULT_MAX_RESULTS,
        "sort": "relevance",
        "full_abstract": False,
    })
    max_results = min(int(opts.get("max_results", DEFAULT_MAX_RESULTS)), MAX_ARXIV_RESULTS)
    sort_by = opts.get("sort", "relevance")
    category = opts.get("category")  # e.g., "cs.AI", "physics"
    full_abstract = opts.get("full_abstract", False)

    if not search_query:
        return "Error: No search query provided."

    # Add category filter to query if specified
    if category:
        search_query = f"cat:{category} AND {search_query}"

    try:
        papers = await async_search_arxiv(search_query, max_results, sort_by)

        if not papers:
            return f"No academic papers found for '{search_query}'"

        # Format the results
        sort_desc = "newest first" if sort_by == "date" else "most relevant"
        results = [f"Found {len(papers)} papers on ArXiv ({sort_desc}):\n"]

        for i, paper in enumerate(papers, 1):
            # Get first 3 authors (some papers have dozens)
            authors = ", ".join([author.name for author in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += f" et al. ({len(paper.authors)} total)"

            # Get summary
            summary = paper.summary.replace('\n', ' ')
            if not full_abstract:
                summary = truncate(summary, ABSTRACT_MAX_CHARS)

            # Get categories
            categories = ", ".join(paper.categories[:3])

            # Format each paper entry
            results.append(
                f"{i}. **{paper.title}**\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   Categories: {categories}\n"
                f"   URL: {paper.entry_id}\n"
                f"   Abstract: {summary}\n"
            )

        return "\n".join(results)

    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


# Create the LangChain Tool wrapper
arxiv_tool = create_tool(
    "arxiv_search",
    search_arxiv,
    "Search ArXiv for PRE-PRINTS — the latest unpublished research in STEM fields. "
    "ArXiv papers are first-to-publish but NOT peer-reviewed."
    "\n\nUSE FOR:"
    "\n- Cutting-edge research: newest papers in AI, ML, physics, math, CS, statistics"
    "\n- Free full-text PDFs of recent papers"
    "\n- Filtering by arXiv category (cs.AI, cs.LG, physics, math, stat.ML, etc.)"
    "\n\nDO NOT USE FOR:"
    "\n- Peer-reviewed/published papers (use google_scholar — it covers journals)"
    "\n- Non-STEM fields: medicine, history, humanities, social science (use google_scholar)"
    "\n\nSIMPLE: 'transformer neural networks' | "
    "ADVANCED: {\"query\": \"attention\", \"max_results\": 10, \"sort\": \"date\", \"category\": \"cs.AI\"}"
    "\n\nRULE: Need the LATEST pre-prints in STEM? -> arxiv. "
    "Need PUBLISHED, peer-reviewed papers? -> google_scholar.",
)

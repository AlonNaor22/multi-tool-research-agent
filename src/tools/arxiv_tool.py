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

import json
import arxiv
from langchain_core.tools import Tool
from src.utils import retry_on_error


# Configuration
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 15
DEFAULT_SUMMARY_LENGTH = 400

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


@retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
def search_arxiv(query: str) -> str:
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
    max_results = DEFAULT_MAX_RESULTS
    sort_by = "relevance"
    category = None
    full_abstract = False

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            search_query = options.get("query", "")
            max_results = min(options.get("max_results", DEFAULT_MAX_RESULTS), MAX_ALLOWED_RESULTS)
            sort_by = options.get("sort", "relevance")
            category = options.get("category")  # e.g., "cs.AI", "physics"
            full_abstract = options.get("full_abstract", False)
        else:
            search_query = query
    except json.JSONDecodeError:
        search_query = query

    if not search_query:
        return "Error: No search query provided."

    # Add category filter to query if specified
    if category:
        search_query = f"cat:{category} AND {search_query}"

    try:
        # Create a client to interact with ArXiv API
        client = arxiv.Client()

        # Determine sort criterion
        if sort_by == "date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        else:
            sort_criterion = arxiv.SortCriterion.Relevance

        # Create a search query
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion
        )

        # Execute the search and collect results
        papers = list(client.results(search))

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
                if len(summary) > DEFAULT_SUMMARY_LENGTH:
                    summary = summary[:DEFAULT_SUMMARY_LENGTH] + "..."

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
arxiv_tool = Tool(
    name="arxiv_search",
    func=search_arxiv,
    description=(
        "Search for academic papers on ArXiv. Use this for scholarly research, "
        "scientific papers, or peer-reviewed sources on physics, math, CS, AI, ML, etc. "
        "\n\nSIMPLE USAGE: Just provide a topic: 'transformer neural networks'"
        "\n\nADVANCED USAGE: Provide JSON with options:"
        '\n{"query": "attention mechanism", "max_results": 10, "sort": "date", "category": "cs.AI"}'
        "\n\nOPTIONS:"
        "\n- query: Search terms"
        "\n- max_results: 1-15 (default 5)"
        "\n- sort: 'relevance' (default) or 'date' (newest first)"
        "\n- category: Filter by ArXiv category (cs.AI, cs.LG, cs.CL, cs.CV, physics, math, etc.)"
        "\n- full_abstract: true to show complete abstract"
        "\n\nRETURNS: Paper titles, authors, dates, categories, URLs, and abstracts."
    )
)

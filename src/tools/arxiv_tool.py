"""ArXiv academic paper search tool for the research agent.

ArXiv (arxiv.org) is a free repository of academic papers in physics,
mathematics, computer science, and more. This tool allows the agent
to search for academic papers and retrieve their metadata.

KEY CONCEPT: We're using the 'arxiv' Python library which wraps ArXiv's API.
No API key is required - ArXiv is completely free and open.
"""

import arxiv
from langchain_core.tools import Tool
from src.utils import retry_on_error


# Configuration
MAX_RESULTS = 5  # Number of papers to return per search


@retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
def search_arxiv(query: str) -> str:
    """
    Search ArXiv for academic papers.

    HOW THIS WORKS:
    1. We create an arxiv.Client() - this handles HTTP requests to ArXiv's API
    2. We create a Search object with our query and parameters
    3. We iterate through results and format them nicely

    Args:
        query: Search query (e.g., "transformer neural networks", "quantum computing")

    Returns:
        Formatted string with paper titles, authors, and summaries.
    """
    try:
        # Create a client to interact with ArXiv API
        client = arxiv.Client()

        # Create a search query
        # SortCriterion.Relevance = most relevant first
        # SortCriterion.SubmittedDate = newest first
        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # Execute the search and collect results
        papers = list(client.results(search))

        if not papers:
            return f"No academic papers found for '{query}'"

        # Format the results
        results = []
        for i, paper in enumerate(papers, 1):
            # Get first 3 authors (some papers have dozens)
            authors = ", ".join([author.name for author in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += f" et al. ({len(paper.authors)} total)"

            # Truncate summary to ~300 chars for readability
            summary = paper.summary.replace('\n', ' ')[:300]
            if len(paper.summary) > 300:
                summary += "..."

            # Format each paper entry
            results.append(
                f"{i}. **{paper.title}**\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   URL: {paper.entry_id}\n"
                f"   Summary: {summary}\n"
            )

        return f"Found {len(papers)} papers on ArXiv:\n\n" + "\n".join(results)

    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


# Create the LangChain Tool wrapper
# The 'description' is CRUCIAL - this is what the agent reads to decide
# when to use this tool. Be specific and give examples!
arxiv_tool = Tool(
    name="arxiv_search",
    func=search_arxiv,
    description=(
        "Search for academic papers on ArXiv. Use this when the user asks about "
        "academic research, scientific papers, scholarly articles, or wants to find "
        "peer-reviewed sources on topics like physics, mathematics, computer science, "
        "machine learning, AI, quantum computing, etc. "
        "Input should be a search query (e.g., 'attention mechanism transformers', "
        "'reinforcement learning robotics', 'quantum error correction')."
    )
)

"""Web search tool for the research agent.

Uses DuckDuckGo for free web searches without requiring an API key.
This gives the agent access to current information from the internet.

Features:
- Structured results with title, URL, and snippet
- Configurable result count
- Region filtering support
- Retry logic for reliability
"""

import json
from duckduckgo_search import DDGS
from langchain_core.tools import Tool
from src.utils import retry_on_error


# Configuration defaults
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10


@retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    The query can be a simple search string, or a JSON object with options:
    - Simple: "latest AI news"
    - With options: {"query": "latest AI news", "max_results": 5, "region": "us-en"}

    Args:
        query: Search query string or JSON with options

    Returns:
        Formatted search results with title, URL, and snippet.
    """
    # Parse input - could be simple string or JSON with options
    max_results = DEFAULT_MAX_RESULTS
    region = None  # None = worldwide

    try:
        # Try to parse as JSON for advanced options
        if query.strip().startswith("{"):
            options = json.loads(query)
            search_query = options.get("query", "")
            max_results = min(options.get("max_results", DEFAULT_MAX_RESULTS), MAX_ALLOWED_RESULTS)
            region = options.get("region")  # e.g., "us-en", "uk-en", "de-de"
        else:
            search_query = query
    except json.JSONDecodeError:
        search_query = query

    if not search_query:
        return "Error: No search query provided."

    try:
        # Create DuckDuckGo Search instance
        ddgs = DDGS()

        # Perform the search
        search_kwargs = {
            "keywords": search_query,
            "max_results": max_results,
        }

        if region:
            search_kwargs["region"] = region

        results = list(ddgs.text(**search_kwargs))

        if not results:
            return f"No search results found for '{search_query}'"

        # Format results in a structured way
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", result.get("link", "No URL"))
            snippet = result.get("body", result.get("snippet", "No description"))

            # Truncate snippet if too long
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."

            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   URL: {url}\n"
                f"   {snippet}"
            )

        header = f"Found {len(results)} results for '{search_query}':\n"
        return header + "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


# Create the LangChain Tool wrapper
search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Search the web for current information using DuckDuckGo. Use this for "
        "up-to-date information, recent events, or anything not in Wikipedia. "
        "\n\nSIMPLE USAGE: Just provide a search query string."
        "\n\nADVANCED USAGE: Provide JSON with options:"
        '\n{"query": "search terms", "max_results": 5, "region": "us-en"}'
        "\n\nREGION CODES: us-en, uk-en, de-de, fr-fr, es-es, it-it, jp-jp, etc."
        "\n\nRETURNS: Structured results with title, URL, and snippet for each result."
        "\n\nEXAMPLES:"
        "\n- 'Tesla stock price 2024'"
        "\n- 'latest AI developments'"
        '\n- {"query": "climate news", "max_results": 3, "region": "uk-en"}'
    )
)

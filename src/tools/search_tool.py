"""Web search tool for the research agent.

Uses DuckDuckGo for free web searches without requiring an API key.
This gives the agent access to current information from the internet.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


# Use LangChain's built-in DuckDuckGo wrapper
# This handles the integration properly
_ddg_search = DuckDuckGoSearchRun()


def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query (e.g., "latest news on AI")

    Returns:
        Search results as text, or an error message.
    """
    try:
        result = _ddg_search.run(query)
        if not result:
            return f"No search results found for '{query}'"
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# Create the LangChain Tool wrapper
search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Search the web for current information. Use this when you need "
        "up-to-date information, recent news, current events, or anything "
        "that might not be in Wikipedia or requires recent data. "
        "Input should be a search query (e.g., 'Tesla stock price 2024', "
        "'latest AI developments', 'weather in New York')."
    )
)

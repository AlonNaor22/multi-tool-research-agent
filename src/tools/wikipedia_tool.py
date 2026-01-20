"""Wikipedia tool for the research agent.

Uses LangChain's WikipediaAPIWrapper to fetch encyclopedic information.
Great for background knowledge, definitions, and historical facts.
"""

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

# Initialize the Wikipedia wrapper
# top_k_results: How many Wikipedia pages to consider
# doc_content_chars_max: Maximum characters to return (prevents huge responses)
wikipedia_api = WikipediaAPIWrapper(
    top_k_results=1,           # Usually the first result is most relevant
    doc_content_chars_max=2000  # Limit response size
)


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.

    Args:
        query: The topic to search for (e.g., "Python programming language")

    Returns:
        A summary of the Wikipedia article, or an error message.
    """
    try:
        result = wikipedia_api.run(query)
        if not result:
            return f"No Wikipedia article found for '{query}'"
        return result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# Create the LangChain Tool wrapper
wikipedia_tool = Tool(
    name="wikipedia",
    func=search_wikipedia,
    description=(
        "Look up encyclopedic information on Wikipedia. Use this for background "
        "knowledge, definitions, historical facts, or general information about "
        "well-known topics. Input should be a topic or search term "
        "(e.g., 'Albert Einstein', 'Climate change', 'Python programming language')."
    )
)

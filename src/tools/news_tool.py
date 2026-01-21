"""News search tool for the research agent.

Uses DuckDuckGo News search to find recent news articles.
No API key required - completely free to use.

This gives the agent access to current news and recent events.
"""

from duckduckgo_search import DDGS
from langchain_core.tools import Tool
from src.utils import retry_on_error


@retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
def search_news(query: str) -> str:
    """
    Search for recent news articles using DuckDuckGo News.

    Args:
        query: The news topic to search for (e.g., "AI developments", "climate change")

    Returns:
        Formatted news results with titles, sources, and snippets.
    """
    try:
        # Create DuckDuckGo Search instance
        ddgs = DDGS()

        # Search for news articles
        # max_results: How many articles to return
        # timelimit: 'd' = past day, 'w' = past week, 'm' = past month
        results = ddgs.news(
            query,
            max_results=5,
            timelimit='w'  # Past week for recent but not too narrow
        )

        # Convert generator to list
        results_list = list(results)

        if not results_list:
            return f"No recent news found for '{query}'. Try broader search terms."

        # Format the results nicely
        formatted_results = []
        for i, article in enumerate(results_list, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            date = article.get('date', 'Unknown date')
            body = article.get('body', 'No description')
            url = article.get('url', '')

            # Truncate body if too long
            if len(body) > 200:
                body = body[:200] + "..."

            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   Source: {source} | Date: {date}\n"
                f"   {body}\n"
                f"   URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error searching news: {str(e)}"


# Create the LangChain Tool wrapper
news_tool = Tool(
    name="news_search",
    func=search_news,
    description=(
        "Search for recent news articles on a topic. Use this when the user asks "
        "about current events, recent developments, breaking news, or anything that "
        "happened recently (within the past week). "
        "Input should be a news topic (e.g., 'artificial intelligence', 'stock market', "
        "'climate summit', 'tech layoffs'). Returns article titles, sources, and summaries."
    )
)

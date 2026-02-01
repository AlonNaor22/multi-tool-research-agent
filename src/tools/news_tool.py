"""News search tool for the research agent.

Uses DuckDuckGo News search to find recent news articles.
No API key required - completely free to use.

Features:
- Configurable time range (day, week, month)
- Configurable result count
- Region filtering
- Structured output with title, source, date, and snippet
"""

import json
from duckduckgo_search import DDGS
from langchain_core.tools import Tool
from src.utils import retry_on_error


# Configuration
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10
DEFAULT_TIMELIMIT = "w"  # Past week


@retry_on_error(max_retries=2, delay=2.0, exceptions=(Exception,))
def search_news(query: str) -> str:
    """
    Search for recent news articles using DuckDuckGo News.

    Input can be:
    - Simple topic string: "AI developments"
    - JSON with options: {"query": "AI", "timelimit": "d", "max_results": 10}

    Args:
        query: News topic string or JSON with options

    Returns:
        Formatted news results with titles, sources, and snippets.
    """
    # Parse input
    max_results = DEFAULT_MAX_RESULTS
    timelimit = DEFAULT_TIMELIMIT
    region = None

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            search_query = options.get("query", "")
            max_results = min(options.get("max_results", DEFAULT_MAX_RESULTS), MAX_ALLOWED_RESULTS)
            timelimit = options.get("timelimit", DEFAULT_TIMELIMIT)
            region = options.get("region")  # e.g., "us-en", "uk-en"
        else:
            search_query = query
    except json.JSONDecodeError:
        search_query = query

    if not search_query:
        return "Error: No search query provided."

    # Validate timelimit
    if timelimit not in ("d", "w", "m"):
        timelimit = DEFAULT_TIMELIMIT

    try:
        ddgs = DDGS()

        # Build search parameters
        search_kwargs = {
            "keywords": search_query,
            "max_results": max_results,
            "timelimit": timelimit
        }

        if region:
            search_kwargs["region"] = region

        results = list(ddgs.news(**search_kwargs))

        if not results:
            time_desc = {"d": "past day", "w": "past week", "m": "past month"}[timelimit]
            return f"No news found for '{search_query}' in the {time_desc}. Try broader terms or a longer time range."

        # Format the results
        time_desc = {"d": "day", "w": "week", "m": "month"}[timelimit]
        formatted_results = [f"Found {len(results)} news articles for '{search_query}' (past {time_desc}):\n"]

        for i, article in enumerate(results, 1):
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
        "Search for recent news articles on a topic. Use this for current events, "
        "breaking news, or recent developments. "
        "\n\nSIMPLE USAGE: Just provide a topic: 'artificial intelligence', 'stock market'"
        "\n\nADVANCED USAGE: Provide JSON with options:"
        '\n{"query": "climate change", "timelimit": "d", "max_results": 10, "region": "uk-en"}'
        "\n\nOPTIONS:"
        "\n- query: Search terms"
        "\n- timelimit: 'd' (past day), 'w' (past week, default), 'm' (past month)"
        "\n- max_results: 1-10 (default 5)"
        "\n- region: 'us-en', 'uk-en', 'de-de', etc."
        "\n\nRETURNS: Article titles, sources, dates, and summaries."
    )
)

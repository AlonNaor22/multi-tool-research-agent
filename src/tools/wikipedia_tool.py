"""Wikipedia tool for the research agent.

Uses the wikipedia Python library to fetch encyclopedic information.
Great for background knowledge, definitions, and historical facts.

Features:
- Configurable result count
- Disambiguation handling
- Summary or full article options
- Search suggestions when exact match not found
"""

import json
import wikipedia
from langchain_core.tools import Tool


# Configuration
DEFAULT_SENTENCES = 5  # Number of sentences in summary
MAX_CHARS = 3000  # Maximum characters to return


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.

    Input can be:
    - Simple topic string: "Python programming"
    - JSON with options: {"query": "Python", "sentences": 10, "suggestion": true}

    Args:
        query: Topic string or JSON with options

    Returns:
        Wikipedia article summary or error message.
    """
    # Parse input
    sentences = DEFAULT_SENTENCES
    auto_suggest = True
    search_results_count = 1

    try:
        if query.strip().startswith("{"):
            options = json.loads(query)
            search_query = options.get("query", "")
            sentences = options.get("sentences", DEFAULT_SENTENCES)
            auto_suggest = options.get("suggestion", True)
            search_results_count = min(options.get("results", 1), 5)
        else:
            search_query = query
    except json.JSONDecodeError:
        search_query = query

    if not search_query:
        return "Error: No search query provided."

    try:
        if search_results_count > 1:
            # Return multiple search results (titles only, for disambiguation)
            search_results = wikipedia.search(search_query, results=search_results_count)

            if not search_results:
                return f"No Wikipedia articles found for '{search_query}'"

            result_parts = [f"Found {len(search_results)} Wikipedia articles for '{search_query}':\n"]

            for i, title in enumerate(search_results, 1):
                try:
                    # Get a brief summary of each
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    result_parts.append(f"{i}. **{page.title}**\n   {summary}")
                except wikipedia.exceptions.DisambiguationError as e:
                    result_parts.append(f"{i}. **{title}** (disambiguation page with {len(e.options)} options)")
                except wikipedia.exceptions.PageError:
                    result_parts.append(f"{i}. **{title}** (page not found)")
                except Exception:
                    result_parts.append(f"{i}. **{title}**")

            return "\n\n".join(result_parts)

        else:
            # Get single article summary
            try:
                summary = wikipedia.summary(search_query, sentences=sentences, auto_suggest=auto_suggest)

                # Get the actual page to retrieve the title and URL
                page = wikipedia.page(search_query, auto_suggest=auto_suggest)

                # Truncate if too long
                if len(summary) > MAX_CHARS:
                    summary = summary[:MAX_CHARS] + "..."

                return (
                    f"**{page.title}**\n"
                    f"URL: {page.url}\n\n"
                    f"{summary}"
                )

            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages
                options = e.options[:10]  # Limit to first 10 options
                options_list = "\n".join(f"  - {opt}" for opt in options)

                return (
                    f"'{search_query}' is ambiguous. Did you mean one of these?\n\n"
                    f"{options_list}\n\n"
                    f"Try searching with a more specific term, or use:\n"
                    f'{{"query": "specific term", "results": 3}} to see multiple results.'
                )

            except wikipedia.exceptions.PageError:
                # Page not found - try to suggest alternatives
                suggestions = wikipedia.search(search_query, results=5)

                if suggestions:
                    suggestions_list = "\n".join(f"  - {s}" for s in suggestions)
                    return (
                        f"No Wikipedia article found for '{search_query}'.\n\n"
                        f"Did you mean:\n{suggestions_list}"
                    )
                else:
                    return f"No Wikipedia article found for '{search_query}'. Try different search terms."

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# Create the LangChain Tool wrapper
wikipedia_tool = Tool(
    name="wikipedia",
    func=search_wikipedia,
    description=(
        "Look up encyclopedic information on Wikipedia. Use this for background "
        "knowledge, definitions, historical facts, or general information. "
        "\n\nSIMPLE USAGE: Just provide a topic: 'Albert Einstein', 'Climate change'"
        "\n\nADVANCED USAGE: Provide JSON with options:"
        '\n{"query": "Python", "sentences": 10, "results": 3}'
        "\n\nOPTIONS:"
        "\n- query: Search topic"
        "\n- sentences: Number of sentences in summary (default 5)"
        "\n- results: Number of articles to return (1-5, for disambiguation)"
        "\n- suggestion: true/false to enable auto-suggestions (default true)"
        "\n\nHANDLES: Disambiguation pages, suggests alternatives when not found."
    )
)

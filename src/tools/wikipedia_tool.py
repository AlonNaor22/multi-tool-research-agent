"""Wikipedia tool for the research agent.

Uses the wikipedia Python library to fetch encyclopedic information.
Great for background knowledge, definitions, and historical facts.

Features:
- Configurable result count
- Disambiguation handling
- Summary or full article options
- Search suggestions when exact match not found
"""

import wikipedia
from src.utils import async_run_with_timeout, create_tool, parse_tool_input, truncate
from src.constants import DEFAULT_SEARCH_TIMEOUT, WIKI_MAX_CHARS


# Configuration
DEFAULT_SENTENCES = 5  # Number of sentences in summary


async def search_wikipedia(query: str) -> str:
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
    search_query, opts = parse_tool_input(query, {
        "sentences": DEFAULT_SENTENCES,
        "suggestion": True,
        "results": 1,
    })
    sentences = opts.get("sentences", DEFAULT_SENTENCES)
    auto_suggest = opts.get("suggestion", True)
    search_results_count = min(int(opts.get("results", 1)), 5)

    if not search_query:
        return "Error: No search query provided."

    try:
        if search_results_count > 1:
            # Return multiple search results (titles only, for disambiguation).
            # The wikipedia library has no timeout parameter, so we wrap the call.
            search_results = await async_run_with_timeout(
                lambda: wikipedia.search(search_query, results=search_results_count),
                timeout=DEFAULT_SEARCH_TIMEOUT,
            )

            if not search_results:
                return f"No Wikipedia articles found for '{search_query}'"

            result_parts = [f"Found {len(search_results)} Wikipedia articles for '{search_query}':\n"]

            for i, title in enumerate(search_results, 1):
                try:
                    # Get a brief summary of each (with timeout protection)
                    page = await async_run_with_timeout(
                        lambda t=title: wikipedia.page(t, auto_suggest=False),
                        timeout=DEFAULT_SEARCH_TIMEOUT,
                    )
                    summary = await async_run_with_timeout(
                        lambda t=title: wikipedia.summary(t, sentences=2, auto_suggest=False),
                        timeout=DEFAULT_SEARCH_TIMEOUT,
                    )
                    summary = truncate(summary, 200)
                    result_parts.append(f"{i}. **{page.title}**\n   {summary}")
                except wikipedia.exceptions.DisambiguationError as e:
                    result_parts.append(f"{i}. **{title}** (disambiguation page with {len(e.options)} options)")
                except wikipedia.exceptions.PageError:
                    result_parts.append(f"{i}. **{title}** (page not found)")
                except Exception as e:
                    # Catch unexpected errors (network, parsing) so one failed
                    # result doesn't abort the entire multi-result search.
                    result_parts.append(f"{i}. **{title}** (error: {str(e)[:80]})")

            return "\n\n".join(result_parts)

        else:
            # Get single article summary
            try:
                # Wrap calls with timeout -- wikipedia library has no timeout parameter
                summary = await async_run_with_timeout(
                    lambda: wikipedia.summary(search_query, sentences=sentences, auto_suggest=auto_suggest),
                    timeout=DEFAULT_SEARCH_TIMEOUT,
                )

                # Get the actual page to retrieve the title and URL
                page = await async_run_with_timeout(
                    lambda: wikipedia.page(search_query, auto_suggest=auto_suggest),
                    timeout=DEFAULT_SEARCH_TIMEOUT,
                )

                # Truncate if too long
                summary = truncate(summary, WIKI_MAX_CHARS)

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
                # Page not found -- try to suggest alternatives
                suggestions = await async_run_with_timeout(
                    lambda: wikipedia.search(search_query, results=5),
                    timeout=DEFAULT_SEARCH_TIMEOUT,
                )

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
wikipedia_tool = create_tool(
    "wikipedia",
    search_wikipedia,
    "Look up EXPLANATIONS, HISTORY, and CONTEXT on Wikipedia. Use for understanding "
    "topics, not for getting specific numbers. "
    "\n\nUSE FOR:"
    "\n- What is something: 'What is machine learning', 'Python programming'"
    "\n- History/background: 'History of the Internet', 'Albert Einstein biography'"
    "\n- Concepts explained: 'How does DNA work', 'What caused World War 2'"
    "\n- General knowledge: 'Climate change', 'Renaissance art'"
    "\n\nDO NOT USE FOR:"
    "\n- Specific numbers/measurements (use wolfram_alpha)"
    "\n- Current events (use web_search)"
    "\n\nSIMPLE: 'Albert Einstein' | ADVANCED: {\"query\": \"Python\", \"sentences\": 10}"
    "\n\nRULE: Need an EXPLANATION? -> Wikipedia. Need a NUMBER? -> Wolfram.",
)

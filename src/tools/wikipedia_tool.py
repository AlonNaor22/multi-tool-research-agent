"""Wikipedia lookup tool."""

import wikipedia as _wikipedia_lib
from langchain_core.tools import tool
from src.utils import (
    async_run_with_timeout, parse_tool_input, truncate,
    safe_tool_call, require_input,
)
from src.constants import DEFAULT_SEARCH_TIMEOUT, WIKI_MAX_CHARS

# ─── Module overview ───────────────────────────────────────────────
# Looks up Wikipedia article summaries and search results using
# the wikipedia library. Handles disambiguation and suggestions.
# ───────────────────────────────────────────────────────────────────

# Configuration
DEFAULT_SENTENCES = 5  # Number of sentences in summary


# Takes a query string (or JSON with sentences/results options).
# Returns article summary with title/URL, or multiple search results.
@safe_tool_call("searching Wikipedia")
async def wikipedia(query: str) -> str:
    """Look up EXPLANATIONS, HISTORY, and CONTEXT on Wikipedia. Use for understanding topics, not for getting specific numbers.

USE FOR:
- What is something: 'What is machine learning', 'Python programming'
- History/background: 'History of the Internet', 'Albert Einstein biography'
- Concepts explained: 'How does DNA work', 'What caused World War 2'
- General knowledge: 'Climate change', 'Renaissance art'

DO NOT USE FOR:
- Specific numbers/measurements (use wolfram_alpha)
- Current events (use web_search)

SIMPLE: 'Albert Einstein' | ADVANCED: {"query": "Python", "sentences": 10}

RULE: Need an EXPLANATION? -> Wikipedia. Need a NUMBER? -> Wolfram."""
    # Parse input
    search_query, opts = parse_tool_input(query, {
        "sentences": DEFAULT_SENTENCES,
        "suggestion": True,
        "results": 1,
    })
    sentences = opts.get("sentences", DEFAULT_SENTENCES)
    auto_suggest = opts.get("suggestion", True)
    search_results_count = min(int(opts.get("results", 1)), 5)

    err = require_input(search_query, "search query")
    if err: return err

    if search_results_count > 1:
        # Return multiple search results (titles only, for disambiguation).
        # The wikipedia library has no timeout parameter, so we wrap the call.
        search_results = await async_run_with_timeout(
            lambda: _wikipedia_lib.search(search_query, results=search_results_count),
            timeout=DEFAULT_SEARCH_TIMEOUT,
        )

        if not search_results:
            return f"No Wikipedia articles found for '{search_query}'"

        result_parts = [f"Found {len(search_results)} Wikipedia articles for '{search_query}':\n"]

        for i, title in enumerate(search_results, 1):
            try:
                # Get a brief summary of each (with timeout protection)
                page = await async_run_with_timeout(
                    lambda t=title: _wikipedia_lib.page(t, auto_suggest=False),
                    timeout=DEFAULT_SEARCH_TIMEOUT,
                )
                summary = await async_run_with_timeout(
                    lambda t=title: _wikipedia_lib.summary(t, sentences=2, auto_suggest=False),
                    timeout=DEFAULT_SEARCH_TIMEOUT,
                )
                summary = truncate(summary, 200)
                result_parts.append(f"{i}. **{page.title}**\n   {summary}")
            except _wikipedia_lib.exceptions.DisambiguationError as e:
                result_parts.append(f"{i}. **{title}** (disambiguation page with {len(e.options)} options)")
            except _wikipedia_lib.exceptions.PageError:
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
                lambda: _wikipedia_lib.summary(search_query, sentences=sentences, auto_suggest=auto_suggest),
                timeout=DEFAULT_SEARCH_TIMEOUT,
            )

            # Get the actual page to retrieve the title and URL
            page = await async_run_with_timeout(
                lambda: _wikipedia_lib.page(search_query, auto_suggest=auto_suggest),
                timeout=DEFAULT_SEARCH_TIMEOUT,
            )

            # Truncate if too long
            summary = truncate(summary, WIKI_MAX_CHARS)

            return (
                f"**{page.title}**\n"
                f"URL: {page.url}\n\n"
                f"{summary}"
            )

        except _wikipedia_lib.exceptions.DisambiguationError as e:
            # Handle disambiguation pages
            options = e.options[:10]  # Limit to first 10 options
            options_list = "\n".join(f"  - {opt}" for opt in options)

            return (
                f"'{search_query}' is ambiguous. Did you mean one of these?\n\n"
                f"{options_list}\n\n"
                f"Try searching with a more specific term, or use:\n"
                f'{{"query": "specific term", "results": 3}} to see multiple results.'
            )

        except _wikipedia_lib.exceptions.PageError:
            # Page not found -- try to suggest alternatives
            suggestions = await async_run_with_timeout(
                lambda: _wikipedia_lib.search(search_query, results=5),
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


wikipedia_tool = tool(wikipedia)

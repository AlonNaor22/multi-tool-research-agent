"""Wikipedia lookup tool."""

import asyncio
from typing import Type

import wikipedia as _wikipedia_lib
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.utils import (
    async_run_with_timeout, truncate, safe_tool_call, require_input,
)
from src.constants import DEFAULT_SEARCH_TIMEOUT, WIKI_MAX_CHARS

# ─── Module overview ───────────────────────────────────────────────
# Looks up Wikipedia article summaries and search results using the
# wikipedia library. Schema is enforced via args_schema.
# ───────────────────────────────────────────────────────────────────

DEFAULT_SENTENCES = 5  # Number of sentences in summary
MAX_RESULTS = 5        # Upper bound when listing multiple results


# Takes (query, sentences, suggestion, results). Returns article summary
# with title/URL, or multiple search results when results > 1.
@safe_tool_call("searching Wikipedia")
async def wikipedia(
    query: str,
    sentences: int = DEFAULT_SENTENCES,
    suggestion: bool = True,
    results: int = 1,
) -> str:
    """Look up Wikipedia articles and return a formatted summary or result list."""
    err = require_input(query, "search query")
    if err:
        return err

    results = min(int(results), MAX_RESULTS)

    if results > 1:
        # Return multiple search results (titles only, for disambiguation).
        # The wikipedia library has no timeout parameter, so we wrap the call.
        search_results = await async_run_with_timeout(
            lambda: _wikipedia_lib.search(query, results=results),
            timeout=DEFAULT_SEARCH_TIMEOUT,
        )

        if not search_results:
            return f"No Wikipedia articles found for '{query}'"

        result_parts = [f"Found {len(search_results)} Wikipedia articles for '{query}':\n"]

        for i, title in enumerate(search_results, 1):
            try:
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

    try:
        summary = await async_run_with_timeout(
            lambda: _wikipedia_lib.summary(query, sentences=sentences, auto_suggest=suggestion),
            timeout=DEFAULT_SEARCH_TIMEOUT,
        )
        page = await async_run_with_timeout(
            lambda: _wikipedia_lib.page(query, auto_suggest=suggestion),
            timeout=DEFAULT_SEARCH_TIMEOUT,
        )

        summary = truncate(summary, WIKI_MAX_CHARS)

        return (
            f"**{page.title}**\n"
            f"URL: {page.url}\n\n"
            f"{summary}"
        )

    except _wikipedia_lib.exceptions.DisambiguationError as e:
        options = e.options[:10]
        options_list = "\n".join(f"  - {opt}" for opt in options)

        return (
            f"'{query}' is ambiguous. Did you mean one of these?\n\n"
            f"{options_list}\n\n"
            f"Try a more specific term, or set results=3 to see multiple summaries."
        )

    except _wikipedia_lib.exceptions.PageError:
        suggestions = await async_run_with_timeout(
            lambda: _wikipedia_lib.search(query, results=5),
            timeout=DEFAULT_SEARCH_TIMEOUT,
        )

        if suggestions:
            suggestions_list = "\n".join(f"  - {s}" for s in suggestions)
            return (
                f"No Wikipedia article found for '{query}'.\n\n"
                f"Did you mean:\n{suggestions_list}"
            )
        return f"No Wikipedia article found for '{query}'. Try different search terms."


class WikipediaInput(BaseModel):
    """Inputs for the wikipedia tool."""
    query: str = Field(description="Wikipedia article title or search term.")
    sentences: int = Field(
        default=DEFAULT_SENTENCES,
        ge=1,
        le=20,
        description=f"Number of sentences in the summary (1-20). Default {DEFAULT_SENTENCES}.",
    )
    suggestion: bool = Field(
        default=True,
        description="Whether Wikipedia should auto-suggest corrections for the query.",
    )
    results: int = Field(
        default=1,
        ge=1,
        le=MAX_RESULTS,
        description=(
            f"Number of search results to return (1-{MAX_RESULTS}). "
            "Use >1 for disambiguation; default 1 returns a single article."
        ),
    )


class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = (
        "Look up EXPLANATIONS, HISTORY, and CONTEXT on Wikipedia. Use for understanding "
        "topics, not for getting specific numbers."
        "\n\nUSE FOR:"
        "\n- What is something: 'What is machine learning', 'Python programming'"
        "\n- History/background: 'History of the Internet', 'Albert Einstein biography'"
        "\n- Concepts explained: 'How does DNA work', 'What caused World War 2'"
        "\n- General knowledge: 'Climate change', 'Renaissance art'"
        "\n\nDO NOT USE FOR:"
        "\n- Specific numbers/measurements (use wolfram_alpha)"
        "\n- Current events (use web_search)"
        "\n\nRULE: Need an EXPLANATION? -> Wikipedia. Need a NUMBER? -> Wolfram."
    )
    args_schema: Type[BaseModel] = WikipediaInput

    # Forwards every validated parameter to wikipedia().
    async def _arun(self, **kwargs) -> str:
        return await wikipedia(**kwargs)

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))


wikipedia_tool = WikipediaTool()

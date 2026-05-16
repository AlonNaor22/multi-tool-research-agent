"""Tool modules for the research agent.

Each tool wraps an external service or capability as a LangChain Tool object.
Import individual tools by name from their modules, e.g.:
    from src.tools.search_tool import search_tool

Tools are NOT eagerly imported here to avoid side effects during testing
(test mocks must be installed before tool modules load their dependencies).
"""

__all__ = [
    "calculator_tool",
    "unit_converter_tool",
    "equation_solver_tool",
    "currency_tool",
    "wolfram_tool",
    "wikipedia_tool",
    "search_tool",
    "news_tool",
    "arxiv_tool",
    "youtube_tool",
    "google_scholar_tool",
    "url_tool",
    "pdf_tool",
    "python_repl_tool",
    "visualization_tool",
    "parallel_tool",
    "weather_tool",
    "translation_tool",
    "reddit_tool",
    "wikidata_tool",
]

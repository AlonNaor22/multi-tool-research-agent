"""Shared constants used across multiple tools.

Centralizes values that were previously duplicated in individual tool files:
User-Agent strings, HTTP timeouts, content size limits, API URLs, and chart defaults.
"""

# ---------------------------------------------------------------------------
# HTTP headers — used by tools that make direct HTTP requests
# (youtube, google_scholar, pdf, url, etc.)
# ---------------------------------------------------------------------------
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_HTTP_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------
DEFAULT_HTTP_TIMEOUT = 15       # For standard HTTP requests (url, youtube, scholar)
DEFAULT_SEARCH_TIMEOUT = 30     # For search APIs that lack their own timeout parameter
PARALLEL_SEARCH_TIMEOUT = 60    # Collective wall-clock limit for parallel_search

# ---------------------------------------------------------------------------
# Content size limits (characters)
# ---------------------------------------------------------------------------
DEFAULT_MAX_CONTENT_CHARS = 5000    # Default truncation limit for tool output
MAX_OUTPUT_LENGTH = 10000           # Python REPL output cap

# ---------------------------------------------------------------------------
# Chart rendering defaults (visualization_tool)
# ---------------------------------------------------------------------------
CHART_FIGSIZE = (10, 6)    # Width x height in inches
CHART_DPI = 150             # Dots per inch — produces 1500x900 px images

# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------
# When truncating at a sentence/line boundary, only use the boundary if it
# preserves at least this fraction of the target length. Avoids cutting too
# aggressively when the nearest boundary is far from the limit.
TRUNCATION_PRESERVE_RATIO = 0.7

# ---------------------------------------------------------------------------
# API URLs
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
DEFAULT_CACHE_TTL = 300  # Seconds before a cached result expires (5 minutes)

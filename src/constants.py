"""Shared constants used across multiple tools and the orchestration layer.

Centralizes values that were previously duplicated in individual tool files:
User-Agent strings, HTTP timeouts, content size limits, API URLs, chart
defaults, specialist names, event types, research modes, and step statuses.
"""

from typing import Literal

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
CSV_MAX_OUTPUT_CHARS = 8000         # CSV/spreadsheet output cap

# ---------------------------------------------------------------------------
# Search result defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_RESULTS = 5             # Default number of search results
MAX_SEARCH_RESULTS = 10             # Upper bound for most search tools
MAX_ARXIV_RESULTS = 15              # ArXiv allows more results

# ---------------------------------------------------------------------------
# Per-domain truncation limits (characters)
# ---------------------------------------------------------------------------
SNIPPET_MAX_CHARS = 250             # Search result snippets
ARTICLE_BODY_MAX_CHARS = 200        # News article body previews
ABSTRACT_MAX_CHARS = 400            # Academic paper abstracts
DESCRIPTION_MAX_CHARS = 200         # Video/repo descriptions
WIKI_MAX_CHARS = 3000               # Wikipedia article summaries
REDDIT_SELFTEXT_MAX_CHARS = 300     # Reddit post text previews
GITHUB_DESC_MAX_CHARS = 150         # GitHub repo descriptions
WIKIDATA_HEADING_MAX_CHARS = 100    # Wikidata heading text
PDF_MAX_PAGES = 20                  # Max PDF pages to extract

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

# ---------------------------------------------------------------------------
# Specialist names (multi-agent orchestration)
# ---------------------------------------------------------------------------
SPECIALIST_RESEARCH = "research"
SPECIALIST_MATH = "math"
SPECIALIST_ANALYSIS = "analysis"
SPECIALIST_FACT_CHECKER = "fact_checker"
SPECIALIST_TRANSLATION = "translation"

ALL_SPECIALIST_NAMES = (
    SPECIALIST_RESEARCH,
    SPECIALIST_MATH,
    SPECIALIST_ANALYSIS,
    SPECIALIST_FACT_CHECKER,
    SPECIALIST_TRANSLATION,
)

# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------
EVENT_PLAN_CREATED = "plan_created"
EVENT_PHASE_STARTED = "phase_started"
EVENT_SPECIALIST_STARTED = "specialist_started"
EVENT_SPECIALIST_DONE = "specialist_done"
EVENT_PHASE_DONE = "phase_done"
EVENT_SYNTHESIS_TOKEN = "synthesis_token"
EVENT_DONE = "done"
EVENT_STEP_STARTED = "step_started"
EVENT_STEP_TOOL = "step_tool"
EVENT_STEP_DONE = "step_done"

# ---------------------------------------------------------------------------
# Research modes (UI radio selection)
# ---------------------------------------------------------------------------
MODE_AUTO = "Auto"
MODE_DIRECT = "Direct"
MODE_PLAN_EXECUTE = "Plan-and-Execute"
MODE_MULTI_AGENT = "Multi-Agent"

RESEARCH_MODES = (MODE_AUTO, MODE_DIRECT, MODE_PLAN_EXECUTE, MODE_MULTI_AGENT)

# ---------------------------------------------------------------------------
# Step / specialist status (unified vocabulary)
# ---------------------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

StepStatus = Literal["pending", "in_progress", "done"]

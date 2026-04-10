"""Utility functions for the research agent.

Contains async retry logic, timeout wrappers, a shared aiohttp session,
Anthropic content-block helpers, tool-input parsing, text truncation,
tool factory, HTTP fetch helper, caching decorator, and a simple TTL cache.
"""

import asyncio
import json
import re
import time
import functools
import hashlib
from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

import aiohttp
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool

from src.constants import DEFAULT_HTTP_TIMEOUT, DEFAULT_HTTP_HEADERS, DEFAULT_CACHE_TTL


# ---------------------------------------------------------------------------
# Anthropic content helpers
# ---------------------------------------------------------------------------

def flatten_content(content: Union[str, List[dict]], sep: str = " ") -> str:
    """Flatten Anthropic content blocks into a plain string.

    Claude may return ``content`` as either a plain ``str`` or a
    ``list[dict]`` where each dict has ``{"type": "text", "text": "..."}``.
    This function handles both and joins text blocks with *sep*.

    Args:
        content: A string passthrough, or a list of Anthropic content blocks.
        sep: Separator for joining text blocks (default ``" "``).

    Returns:
        The concatenated text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def extract_chunk_text(chunk) -> str:
    """Return the text content from an LLM message or streaming chunk.

    Works with ``AIMessage``, ``AIMessageChunk``, or any object whose
    ``.content`` attribute is a ``str`` or a list of Anthropic content
    blocks.
    """
    return flatten_content(getattr(chunk, "content", ""), sep="")


def extract_ai_answer(result: dict, default: str = "No answer was generated.") -> str:
    """Extract the final text answer from a LangGraph agent result.

    Walks the ``messages`` list in reverse and returns the last
    ``AIMessage`` that contains non-empty text.

    Args:
        result: The dict returned by ``agent.ainvoke({"messages": ...})``.
        default: Fallback string if no text is found.

    Returns:
        The AI's answer text, or *default*.
    """
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            text = flatten_content(msg.content, sep="\n")
            if text:
                return text
    return default


# ---------------------------------------------------------------------------
# Async retry decorator
# ---------------------------------------------------------------------------

async def _retry_sleep(seconds: float) -> None:
    """Patchable wrapper around ``asyncio.sleep`` used only by the retry
    decorator.  Tests can ``patch("src.utils._retry_sleep")`` to zero-out
    retry delays without affecting ``asyncio.sleep`` globally."""
    await asyncio.sleep(seconds)


def async_retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Async decorator that retries a coroutine if it raises an exception.

    Uses asyncio.sleep for non-blocking backoff with rate-limit detection
    (HTTP 429 → 5× longer backoff).

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry on

    Example:
        @async_retry_on_error(max_retries=3, delay=1.0)
        async def call_api():
            async with session.get(url) as resp:
                return await resp.json()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        print(f"  ⚠️  All {max_retries} retries failed for {func.__name__}")
                        raise

                    is_rate_limited = _is_rate_limit_error(e)
                    retry_delay = current_delay * 5 if is_rate_limited else current_delay
                    reason = "rate limited" if is_rate_limited else str(e)[:50]

                    print(f"  🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} "
                          f"after {reason}...")

                    await _retry_sleep(retry_delay)
                    current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception represents an HTTP 429 rate limit error."""
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code == 429
    if hasattr(error, "status"):
        return error.status == 429
    error_str = str(error)
    return "429" in error_str or "rate limit" in error_str.lower()


# ---------------------------------------------------------------------------
# Async timeout wrapper
# ---------------------------------------------------------------------------

async def async_run_with_timeout(func: Callable, args: tuple = (), timeout: int = 30) -> Any:
    """
    Run a blocking function in a thread with a hard async timeout.

    Wraps synchronous/blocking functions (e.g., third-party libraries without
    async support) in asyncio.to_thread() and enforces a timeout via
    asyncio.wait_for().

    Args:
        func: The (sync) function to call.
        args: Positional arguments to pass to the function.
        timeout: Maximum seconds to wait before raising TimeoutError.

    Returns:
        The return value of func(*args).

    Raises:
        TimeoutError: If the function doesn't complete within the timeout.

    Example:
        results = await async_run_with_timeout(
            lambda: list(ddgs.text(query)), timeout=30
        )
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        name = func.__name__ if hasattr(func, "__name__") else "Function"
        raise TimeoutError(f"{name} timed out after {timeout} seconds")


# ---------------------------------------------------------------------------
# Shared aiohttp session
# ---------------------------------------------------------------------------

_session: Optional[aiohttp.ClientSession] = None


async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get or create the shared aiohttp session (lazy initialization)."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_aiohttp_session() -> None:
    """Close the shared aiohttp session (call at shutdown)."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


# ---------------------------------------------------------------------------
# Sync bridge for LangChain Tool.func
# ---------------------------------------------------------------------------

def make_sync(async_fn: Callable) -> Callable:
    """Create a sync wrapper around an async function.

    Needed because LangChain's Tool() constructor requires a sync `func` parameter.
    The agent uses `ainvoke()` which calls the `coroutine` directly — this sync
    wrapper is a fallback that should rarely be called in practice.
    """
    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g., Streamlit) — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, async_fn(*args, **kwargs)).result()
        else:
            return asyncio.run(async_fn(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Safe execute
# ---------------------------------------------------------------------------

async def safe_execute(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """
    Execute an async function and return a default value if it fails.

    Args:
        func: Async function to execute
        *args: Arguments to pass to the function
        default: Value to return if function fails (default: None)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Function result or default value if an error occurred
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️  {func.__name__} failed: {str(e)[:50]}, using default")
        return default


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class TTLCache:
    """Simple in-memory cache with per-entry TTL (time-to-live).

    Avoids redundant API calls for identical queries within a short window.

    Usage:
        cache = TTLCache(ttl=300)
        cache.set("key", value)
        hit = cache.get("key")  # returns value or None if expired
    """

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.time() - timestamp > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp."""
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a deterministic cache key from string parts."""
        raw = "|".join(str(p) for p in parts)
        return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tool-input parsing
# ---------------------------------------------------------------------------

def parse_tool_input(raw: str, defaults: Optional[Dict] = None) -> Tuple[str, Dict]:
    """Parse a tool's raw input string into ``(query, options)``.

    Most tools accept either a plain string or a JSON object with a
    ``"query"`` key plus optional fields.  This replaces the identical
    ``if raw.startswith("{") … json.loads … except JSONDecodeError``
    boilerplate that appeared in 11+ tool files.

    Returns:
        A ``(query_string, options_dict)`` tuple.  If the input is not
        valid JSON, the raw string is returned as the query and *defaults*
        (if any) are used as the options.
    """
    opts: Dict[str, Any] = dict(defaults or {})
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            query = parsed.pop("query", raw)
            opts.update(parsed)
            return query, opts
        except json.JSONDecodeError:
            pass
    return raw, opts


def parse_result_count(
    query: str, default: int = 5, max_allowed: int = 10,
) -> Tuple[str, int]:
    """Extract an optional ``"N results: <query>"`` prefix.

    Several search tools let the user write ``"5 results: quantum computing"``
    to control result count.  This replaces the identical regex that appeared
    in google_scholar, reddit, and youtube tools.

    Returns:
        ``(clean_query, max_results)``
    """
    m = re.match(r"(\d+)\s+results?:\s*(.+)", query, re.IGNORECASE)
    if m:
        return m.group(2).strip(), min(int(m.group(1)), max_allowed)
    return query, default


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Truncate *text* to *limit* characters, appending *suffix* if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_tool(name: str, async_fn: Callable, description: str) -> Tool:
    """Create a LangChain ``Tool`` with sync + async support.

    Replaces the identical ``Tool(name=…, func=make_sync(…), coroutine=…,
    description=…)`` boilerplate found in 17+ tool files.
    """
    return Tool(
        name=name,
        func=make_sync(async_fn),
        coroutine=async_fn,
        description=description,
    )


# ---------------------------------------------------------------------------
# Async HTTP fetch
# ---------------------------------------------------------------------------

async def async_fetch(
    url: str,
    *,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    response_type: str = "json",
) -> Any:
    """Fetch a URL via the shared aiohttp session.

    Wraps the ``get_aiohttp_session() → session.get() → resp.raise_for_status()``
    boilerplate used by 10+ tool files.

    Args:
        url: The URL to GET.
        params: Optional query parameters.
        headers: HTTP headers (defaults to ``DEFAULT_HTTP_HEADERS``).
        timeout: Request timeout in seconds.
        response_type: ``"json"``, ``"text"``, or ``"bytes"``.

    Returns:
        Parsed JSON dict, text string, or raw bytes depending on
        *response_type*.

    Raises:
        aiohttp.ClientError / asyncio.TimeoutError on failure.
    """
    session = await get_aiohttp_session()
    hdrs = headers if headers is not None else dict(DEFAULT_HTTP_HEADERS)
    async with session.get(
        url,
        params=params,
        headers=hdrs,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        if response_type == "json":
            return await resp.json()
        if response_type == "bytes":
            return await resp.read()
        return await resp.text()


# ---------------------------------------------------------------------------
# Caching decorator for tool functions
# ---------------------------------------------------------------------------

def cached_tool(prefix: str, ttl: int = DEFAULT_CACHE_TTL):
    """Decorator that adds TTLCache get/set around a tool's main function.

    Replaces the identical cache boilerplate in google_scholar, reddit,
    youtube, and wikidata tools.

    The cache key is built from ``(prefix, *args)`` — all positional args
    are stringified and hashed.

    Usage::

        _cache = TTLCache(ttl=DEFAULT_CACHE_TTL)

        @cached_tool("scholar")
        async def _fetch_papers(query: str, max_results: int) -> list:
            ...

    The decorator must be applied to the **inner** async function, not the
    top-level tool entry point (which handles input parsing/formatting).
    """
    _cache = TTLCache(ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            key = _cache.make_key(prefix, *(str(a) for a in args))
            cached = _cache.get(key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            _cache.set(key, result)
            return result

        # Expose cache for test clearing
        wrapper._cache = _cache
        return wrapper

    return decorator

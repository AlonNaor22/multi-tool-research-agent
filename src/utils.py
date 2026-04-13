"""Shared utility functions: retry, timeout, HTTP, caching, tool helpers."""

import asyncio
import json
import re
import time
import functools
import hashlib
from typing import Callable, Any, Dict, List, Optional, Tuple, Type, Union

import aiohttp
from langchain_core.messages import AIMessage

from src.constants import DEFAULT_HTTP_TIMEOUT, DEFAULT_HTTP_HEADERS, DEFAULT_CACHE_TTL

# ─── Module overview ───────────────────────────────────────────────
# Shared utilities used across the codebase: retry/timeout wrappers,
# async HTTP helpers, TTL cache, tool-input parsing, text truncation,
# and the safe_tool_call / cached_tool decorators.
# ───────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Anthropic content helpers
# ---------------------------------------------------------------------------

# Takes (content, sep). Normalizes str or Anthropic content-block list to plain text.
def flatten_content(content: Union[str, List[dict]], sep: str = " ") -> str:
    """Takes str or Anthropic content-block list, returns joined plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


# Takes (chunk). Extracts text content from an LLM message or chunk object.
def extract_chunk_text(chunk) -> str:
    """Takes an LLM message/chunk, returns its text content as a string."""
    return flatten_content(getattr(chunk, "content", ""), sep="")


# Takes (result, default). Walks messages in reverse to find the last AIMessage text.
def extract_ai_answer(result: dict, default: str = "No answer was generated.") -> str:
    """Takes agent result dict, returns last AIMessage text or default."""
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
    """Patchable sleep for the retry decorator; tests mock this to skip delays."""
    await asyncio.sleep(seconds)


# Takes (max_retries, delay, backoff, exceptions). Returns a decorator that
# retries an async function with exponential backoff on matching exceptions.
def async_retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator: retries an async fn with exponential backoff on failure."""

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


# Takes (error). Checks status codes and error strings for HTTP 429 indicators.
def _is_rate_limit_error(error: Exception) -> bool:
    """Returns True if the exception looks like an HTTP 429."""
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code == 429
    if hasattr(error, "status"):
        return error.status == 429
    error_str = str(error)
    return "429" in error_str or "rate limit" in error_str.lower()


# ---------------------------------------------------------------------------
# Async timeout wrapper
# ---------------------------------------------------------------------------

# Takes (func, args, timeout). Runs a blocking function in a thread with a timeout.
async def async_run_with_timeout(func: Callable, args: tuple = (), timeout: int = 30) -> Any:
    """Runs a blocking func in a thread; raises TimeoutError after *timeout* seconds."""
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


# Returns the module-level aiohttp session, creating one if needed.
async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Returns the shared aiohttp session, creating it if needed."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


# Closes the shared aiohttp session and resets the module-level reference.
async def close_aiohttp_session() -> None:
    """Closes the shared aiohttp session."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


# ---------------------------------------------------------------------------
# Safe execute
# ---------------------------------------------------------------------------

# Takes (func, *args, default). Calls an async function, returning default on error.
async def safe_execute(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """Calls an async fn; returns *default* on any exception."""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️  {func.__name__} failed: {str(e)[:50]}, using default")
        return default


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class TTLCache:
    """In-memory cache with per-entry TTL; avoids redundant API calls."""

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Returns cached value or None if missing/expired."""
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

# Takes (raw, defaults). Parses raw string or JSON into (query, options_dict).
def parse_tool_input(raw: str, defaults: Optional[Dict] = None) -> Tuple[str, Dict]:
    """Takes raw tool input (string or JSON), returns (query, options_dict)."""
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


# Takes (query, default, max_allowed). Extracts "N results: query" prefix.
# Returns (clean_query, clamped_count).
def parse_result_count(
    query: str, default: int = 5, max_allowed: int = 10,
) -> Tuple[str, int]:
    """Extracts 'N results: query' prefix; returns (clean_query, count)."""
    m = re.match(r"(\d+)\s+results?:\s*(.+)", query, re.IGNORECASE)
    if m:
        return m.group(2).strip(), min(int(m.group(1)), max_allowed)
    return query, default


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------

# Takes (text, limit, suffix). Cuts text to limit chars, appending suffix if truncated.
def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Truncate *text* to *limit* chars, appending *suffix* if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


# Takes (value, label). Returns an error string if value is blank, else None.
def require_input(value: str, label: str = "query") -> Optional[str]:
    """Return an error string if *value* is empty/blank, else None."""
    if not value or not value.strip():
        return f"Error: No {label} provided."
    return None


# Takes (operation). Returns a decorator that wraps an async tool fn in try/except.
def safe_tool_call(operation: str):
    """Decorator that wraps an async tool fn in try/except, returning 'Error …' on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> str:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return f"Error {operation}: {str(e)}"
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Async HTTP fetch
# ---------------------------------------------------------------------------

# Takes (url, params, headers, timeout, response_type). GETs a URL via shared session.
# Returns parsed json, text, or bytes depending on response_type.
async def async_fetch(
    url: str,
    *,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    response_type: str = "json",
) -> Any:
    """GETs a URL via shared aiohttp session; returns json/text/bytes."""
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

# Takes (prefix, ttl). Returns a decorator that caches async fn results by args.
def cached_tool(prefix: str, ttl: int = DEFAULT_CACHE_TTL):
    """Decorator: caches an async fn's return value by (prefix, *args) key."""
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

"""Utility functions for the research agent.

Contains async retry logic, timeout wrappers, a shared aiohttp session,
and a simple TTL cache for reducing redundant API calls.
"""

import asyncio
import time
import functools
import hashlib
from typing import Callable, Any, Dict, Optional, Tuple, Type

import aiohttp


# ---------------------------------------------------------------------------
# Async retry decorator
# ---------------------------------------------------------------------------

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

                    await asyncio.sleep(retry_delay)
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

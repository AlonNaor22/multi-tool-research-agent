"""Utility functions for the research agent.

Contains helper functions like retry logic, timeout wrappers,
and a simple TTL cache for reducing redundant API calls.
"""

import time
import functools
import hashlib
import concurrent.futures
from typing import Callable, Any, Dict, Optional, Tuple, Type


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    A decorator that retries a function if it raises an exception.

    This is useful for network calls that might fail temporarily.

    How Decorators Work:
    --------------------
    A decorator wraps a function to add extra behavior.

    Without decorator:
        result = fetch_data()  # If this fails, it just fails

    With retry decorator:
        @retry_on_error(max_retries=3)
        def fetch_data(): ...

        result = fetch_data()  # If this fails, it retries up to 3 times

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
                 Example: delay=1, backoff=2 → waits 1s, 2s, 4s
        exceptions: Tuple of exception types to catch and retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def call_api():
            response = requests.get("https://api.example.com")
            return response.json()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # Preserves function name and docstring
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            # Try up to max_retries + 1 times (original + retries)
            for attempt in range(max_retries + 1):
                try:
                    # Try to execute the function
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # If we've used all retries, give up
                    if attempt == max_retries:
                        print(f"  ⚠️  All {max_retries} retries failed for {func.__name__}")
                        raise

                    # Detect rate limiting (HTTP 429) and back off more aggressively
                    is_rate_limited = _is_rate_limit_error(e)
                    retry_delay = current_delay * 5 if is_rate_limited else current_delay
                    reason = "rate limited" if is_rate_limited else str(e)[:50]

                    # Log the retry attempt
                    print(f"  🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} "
                          f"after {reason}...")

                    # Wait before retrying (exponential backoff)
                    time.sleep(retry_delay)
                    current_delay *= backoff  # Increase delay for next retry

            # This shouldn't be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception represents an HTTP 429 rate limit error."""
    # requests.exceptions.HTTPError carries a response object
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code == 429
    # Fall back to string matching for other HTTP libraries
    error_str = str(error)
    return "429" in error_str or "rate limit" in error_str.lower()


def safe_execute(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """
    Execute a function and return a default value if it fails.

    This is simpler than retry - just catches errors and returns a fallback.

    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        default: Value to return if function fails (default: None)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Function result or default value if an error occurred

    Example:
        result = safe_execute(risky_function, arg1, arg2, default="fallback")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠️  {func.__name__} failed: {str(e)[:50]}, using default")
        return default


def run_with_timeout(func: Callable, args: tuple = (), timeout: int = 30) -> Any:
    """
    Run a function with a hard timeout using a thread pool.

    Some third-party libraries (arxiv, duckduckgo_search, wikipedia) don't
    expose a timeout parameter. This wrapper ensures they can't block forever
    by running them in a thread and raising TimeoutError if they exceed the limit.

    Args:
        func: The function to call.
        args: Positional arguments to pass to the function.
        timeout: Maximum seconds to wait before raising TimeoutError.

    Returns:
        The return value of func(*args).

    Raises:
        TimeoutError: If the function doesn't complete within the timeout.
        Exception: Any exception raised by func is re-raised.

    Example:
        # Instead of: results = list(ddgs.text(query))  # can hang forever
        results = run_with_timeout(lambda: list(ddgs.text(query)), timeout=30)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"{func.__name__ if hasattr(func, '__name__') else 'Function'} "
                f"timed out after {timeout} seconds"
            )


class TTLCache:
    """Simple in-memory cache with per-entry TTL (time-to-live).

    Avoids redundant API calls for identical queries within a short window.
    Thread-safe for the typical single-threaded agent loop.

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

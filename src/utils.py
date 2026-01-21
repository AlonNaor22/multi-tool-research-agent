"""Utility functions for the research agent.

Contains helper functions like retry logic that can be used across tools.
"""

import time
import functools
from typing import Callable, Any, Tuple, Type


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

                    # Log the retry attempt
                    print(f"  🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} "
                          f"after error: {str(e)[:50]}...")

                    # Wait before retrying (exponential backoff)
                    time.sleep(current_delay)
                    current_delay *= backoff  # Increase delay for next retry

            # This shouldn't be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


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

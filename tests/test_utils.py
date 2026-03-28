"""Tests for src/utils.py — retry decorator, safe_execute, TTLCache, rate limit detection."""

import time
import pytest
from unittest.mock import patch
from src.utils import retry_on_error, safe_execute, TTLCache, _is_rate_limit_error


class TestRetryOnError:
    """Tests for the retry_on_error decorator."""

    def test_succeeds_first_try(self):
        """Function that works should return normally."""
        @retry_on_error(max_retries=3, delay=0)
        def always_works():
            return "success"

        assert always_works() == "success"

    def test_retries_then_succeeds(self):
        """Function that fails once then succeeds should return on retry."""
        call_count = 0

        @retry_on_error(max_retries=3, delay=0)
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("temporary failure")
            return "recovered"

        result = fails_once()
        assert result == "recovered"
        assert call_count == 2

    def test_all_retries_exhausted(self):
        """Function that always fails should raise after max retries."""
        @retry_on_error(max_retries=2, delay=0)
        def always_fails():
            raise ConnectionError("server down")

        with pytest.raises(ConnectionError, match="server down"):
            always_fails()

    def test_only_catches_specified_exceptions(self):
        """Should only retry on the specified exception types."""
        @retry_on_error(max_retries=3, delay=0, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("wrong type")

        # TypeError is not in the exceptions tuple, so it should propagate immediately
        with pytest.raises(TypeError):
            raises_type_error()

    def test_preserves_function_metadata(self):
        """Decorated function should keep its original name and docstring."""
        @retry_on_error(max_retries=1, delay=0)
        def my_function():
            """My docstring."""
            return True

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_passes_arguments_through(self):
        """Arguments should be forwarded to the wrapped function."""
        @retry_on_error(max_retries=1, delay=0)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
        assert add(a=10, b=20) == 30


class TestSafeExecute:
    """Tests for the safe_execute helper."""

    def test_returns_result_on_success(self):
        result = safe_execute(lambda: 42)
        assert result == 42

    def test_returns_default_on_failure(self):
        result = safe_execute(lambda: 1 / 0, default="fallback")
        assert result == "fallback"

    def test_default_is_none(self):
        result = safe_execute(lambda: 1 / 0)
        assert result is None

    def test_passes_args(self):
        result = safe_execute(lambda x, y: x + y, 3, 4)
        assert result == 7


class TestTTLCache:
    """Tests for the TTLCache."""

    def test_set_and_get(self):
        cache = TTLCache(ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_returns_none_for_missing_key(self):
        cache = TTLCache(ttl=60)
        assert cache.get("nonexistent") is None

    def test_expiration(self):
        cache = TTLCache(ttl=0)  # Immediately expires
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = TTLCache(ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_make_key_deterministic(self):
        key1 = TTLCache.make_key("a", "b", "c")
        key2 = TTLCache.make_key("a", "b", "c")
        assert key1 == key2

    def test_make_key_different_inputs(self):
        key1 = TTLCache.make_key("a", "b")
        key2 = TTLCache.make_key("a", "c")
        assert key1 != key2


class TestRateLimitDetection:
    """Tests for rate limit error detection."""

    def test_detects_429_in_string(self):
        error = Exception("HTTP Error 429: Too Many Requests")
        assert _is_rate_limit_error(error) is True

    def test_detects_rate_limit_text(self):
        error = Exception("Rate limit exceeded")
        assert _is_rate_limit_error(error) is True

    def test_normal_error_not_rate_limited(self):
        error = Exception("Connection refused")
        assert _is_rate_limit_error(error) is False

    def test_detects_response_status_429(self):
        class MockResp:
            status_code = 429
        error = Exception("error")
        error.response = MockResp()
        assert _is_rate_limit_error(error) is True

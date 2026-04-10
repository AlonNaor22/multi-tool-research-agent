"""Tests for src/utils.py — async retry, timeout, TTLCache, rate limit detection."""

import time
import asyncio
import pytest
from unittest.mock import patch
from src.utils import (
    async_retry_on_error, async_run_with_timeout,
    safe_execute, TTLCache, _is_rate_limit_error,
)


class TestAsyncRetryOnError:
    """Tests for the async_retry_on_error decorator."""

    async def test_succeeds_first_try(self):
        @async_retry_on_error(max_retries=3, delay=0)
        async def always_works():
            return "success"

        assert await always_works() == "success"

    async def test_retries_then_succeeds(self):
        call_count = 0

        @async_retry_on_error(max_retries=3, delay=0)
        async def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("temporary failure")
            return "recovered"

        result = await fails_once()
        assert result == "recovered"
        assert call_count == 2

    async def test_all_retries_exhausted(self):
        @async_retry_on_error(max_retries=2, delay=0)
        async def always_fails():
            raise ConnectionError("server down")

        with pytest.raises(ConnectionError, match="server down"):
            await always_fails()

    async def test_only_catches_specified_exceptions(self):
        @async_retry_on_error(max_retries=3, delay=0, exceptions=(ValueError,))
        async def raises_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            await raises_type_error()

    async def test_preserves_function_metadata(self):
        @async_retry_on_error(max_retries=1, delay=0)
        async def my_function():
            """My docstring."""
            return True

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    async def test_passes_arguments_through(self):
        @async_retry_on_error(max_retries=1, delay=0)
        async def add(a, b):
            return a + b

        assert await add(2, 3) == 5
        assert await add(a=10, b=20) == 30


class TestAsyncRunWithTimeout:
    """Tests for async_run_with_timeout."""

    async def test_returns_result(self):
        result = await async_run_with_timeout(lambda: 42, timeout=5)
        assert result == 42

    async def test_times_out(self):
        import time as _time
        def slow():
            _time.sleep(5)
            return "done"

        with pytest.raises(TimeoutError):
            await async_run_with_timeout(slow, timeout=0.1)

    async def test_passes_args(self):
        result = await async_run_with_timeout(lambda x: x * 2, args=(21,), timeout=5)
        assert result == 42


class TestSafeExecute:
    """Tests for the async safe_execute helper."""

    async def test_returns_result_on_success(self):
        async def ok():
            return 42
        result = await safe_execute(ok)
        assert result == 42

    async def test_returns_default_on_failure(self):
        async def fail():
            raise ValueError("boom")
        result = await safe_execute(fail, default="fallback")
        assert result == "fallback"

    async def test_default_is_none(self):
        async def fail():
            raise ValueError("boom")
        result = await safe_execute(fail)
        assert result is None


class TestTTLCache:
    """Tests for the TTLCache."""

    async def test_set_and_get(self):
        cache = TTLCache(ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    async def test_returns_none_for_missing_key(self):
        cache = TTLCache(ttl=60)
        assert cache.get("nonexistent") is None

    async def test_expiration(self):
        cache = TTLCache(ttl=0)  # Immediately expires
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    async def test_clear(self):
        cache = TTLCache(ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    async def test_make_key_deterministic(self):
        key1 = TTLCache.make_key("a", "b", "c")
        key2 = TTLCache.make_key("a", "b", "c")
        assert key1 == key2

    async def test_make_key_different_inputs(self):
        key1 = TTLCache.make_key("a", "b")
        key2 = TTLCache.make_key("a", "c")
        assert key1 != key2


class TestRateLimitDetection:
    """Tests for rate limit error detection."""

    async def test_detects_429_in_string(self):
        error = Exception("HTTP Error 429: Too Many Requests")
        assert _is_rate_limit_error(error) is True

    async def test_detects_rate_limit_text(self):
        error = Exception("Rate limit exceeded")
        assert _is_rate_limit_error(error) is True

    async def test_normal_error_not_rate_limited(self):
        error = Exception("Connection refused")
        assert _is_rate_limit_error(error) is False

    async def test_detects_response_status_429(self):
        class MockResp:
            status_code = 429
        error = Exception("error")
        error.response = MockResp()
        assert _is_rate_limit_error(error) is True

    async def test_detects_aiohttp_status_429(self):
        error = Exception("error")
        error.status = 429
        assert _is_rate_limit_error(error) is True

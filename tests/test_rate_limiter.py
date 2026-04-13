"""Tests for src/rate_limiter.py — token budget enforcement."""

import pytest
from src.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_check_budget_passes_when_disabled(self):
        limiter = RateLimiter()
        limiter.tokens_spent = 999_999
        limiter.check_budget()  # Should not raise

    def test_check_budget_passes_under_limit(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=10_000)
        limiter.tokens_spent = 5_000
        limiter.check_budget()  # Should not raise

    def test_check_budget_raises_at_limit(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=10_000)
        limiter.tokens_spent = 10_000

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_budget()

        assert exc_info.value.tokens_spent == 10_000
        assert exc_info.value.budget == 10_000

    def test_check_budget_raises_over_limit(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=5_000)
        limiter.tokens_spent = 7_000

        with pytest.raises(RateLimitExceeded):
            limiter.check_budget()

    def test_record_tokens(self):
        limiter = RateLimiter()
        limiter.record_tokens(500)
        limiter.record_tokens(300)
        assert limiter.tokens_spent == 800

    def test_reset(self):
        limiter = RateLimiter()
        limiter.tokens_spent = 5_000
        limiter.reset()
        assert limiter.tokens_spent == 0

    def test_set_config(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=50_000)
        assert limiter.enabled is True
        assert limiter.budget == 50_000

    def test_set_config_negative_budget(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=-100)
        assert limiter.budget == 0

    def test_tokens_remaining_when_disabled(self):
        limiter = RateLimiter()
        assert limiter.tokens_remaining == -1  # Sentinel for unlimited

    def test_tokens_remaining_when_enabled(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=10_000)
        limiter.tokens_spent = 3_000
        assert limiter.tokens_remaining == 7_000

    def test_tokens_remaining_never_negative(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=5_000)
        limiter.tokens_spent = 8_000
        assert limiter.tokens_remaining == 0

    def test_usage_fraction_partial(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=10_000)
        limiter.tokens_spent = 2_500
        assert limiter.usage_fraction == 0.25

    def test_usage_fraction_capped_at_1(self):
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=5_000)
        limiter.tokens_spent = 10_000
        assert limiter.usage_fraction == 1.0

    def test_rate_limit_exceeded_message(self):
        exc = RateLimitExceeded(tokens_spent=15_000, budget=10_000)
        assert "15,000" in str(exc)
        assert "10,000" in str(exc)

    def test_realtime_config_change(self):
        """Simulate enabling rate limit mid-session from UI."""
        limiter = RateLimiter()
        limiter.record_tokens(5_000)

        # Initially disabled — no limit
        limiter.check_budget()  # OK

        # User enables from UI
        limiter.set_config(enabled=True, budget=3_000)

        # Now exceeds budget
        with pytest.raises(RateLimitExceeded):
            limiter.check_budget()

        # User increases budget
        limiter.set_config(enabled=True, budget=10_000)
        limiter.check_budget()  # OK again

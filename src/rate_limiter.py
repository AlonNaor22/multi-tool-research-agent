"""Rate limiter for the research agent.

Tracks cumulative token usage per session and optionally enforces a budget.
Disabled by default — users can enable and configure the budget in real-time
from the Streamlit UI.
"""


class RateLimitExceeded(Exception):
    """Raised when a query would exceed the token budget."""

    def __init__(self, tokens_spent: int, budget: int):
        self.tokens_spent = tokens_spent
        self.budget = budget
        super().__init__(
            f"Rate limit exceeded: {tokens_spent:,} tokens spent of {budget:,} budget. "
            f"Increase the budget or start a new session."
        )


class RateLimiter:
    """Per-session token budget enforcer.

    Off by default. When enabled, tracks cumulative tokens across queries
    and blocks new queries once the budget is reached.

    Usage:
        limiter = RateLimiter()
        limiter.set_config(enabled=True, budget=50_000)

        limiter.check_budget()        # raises RateLimitExceeded if over budget
        # ... run query ...
        limiter.record_tokens(1500)   # update after query completes
    """

    def __init__(self):
        self.enabled: bool = False
        self.budget: int = 100_000
        self.tokens_spent: int = 0

    def set_config(self, enabled: bool, budget: int) -> None:
        """Update rate limiter settings in real-time (e.g., from UI)."""
        self.enabled = enabled
        self.budget = max(0, budget)

    def check_budget(self) -> None:
        """Raise RateLimitExceeded if the budget is exhausted.

        Call this before starting a new query.
        """
        if self.enabled and self.tokens_spent >= self.budget:
            raise RateLimitExceeded(self.tokens_spent, self.budget)

    def record_tokens(self, tokens: int) -> None:
        """Add tokens from a completed query to the running total."""
        self.tokens_spent += tokens

    def reset(self) -> None:
        """Reset the token counter (e.g., on session clear)."""
        self.tokens_spent = 0

    @property
    def tokens_remaining(self) -> int:
        """Tokens left before the budget is hit. Infinite when disabled."""
        if not self.enabled:
            return -1  # Sentinel for "unlimited"
        return max(0, self.budget - self.tokens_spent)

    @property
    def usage_fraction(self) -> float:
        """Fraction of budget used (0.0–1.0). 0 when disabled."""
        if not self.enabled or self.budget <= 0:
            return 0.0
        return min(1.0, self.tokens_spent / self.budget)

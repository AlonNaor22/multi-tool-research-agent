"""Per-session token budget enforcer."""


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
    """Per-session token budget enforcer with usage tracking."""

    def __init__(self):
        self.enabled: bool = False
        self.budget: int = 100_000
        self.tokens_spent: int = 0

    def set_config(self, enabled: bool, budget: int) -> None:
        self.enabled = enabled
        self.budget = max(0, budget)

    def check_budget(self) -> None:
        if self.enabled and self.tokens_spent >= self.budget:
            raise RateLimitExceeded(self.tokens_spent, self.budget)

    def record_tokens(self, tokens: int) -> None:
        self.tokens_spent += tokens

    def reset(self) -> None:
        self.tokens_spent = 0

    @property
    def tokens_remaining(self) -> int:
        """Return tokens left before budget; -1 when disabled."""
        if not self.enabled:
            return -1
        return max(0, self.budget - self.tokens_spent)

    @property
    def usage_fraction(self) -> float:
        """Return fraction of budget used (0.0 to 1.0); 0.0 when disabled."""
        if not self.enabled or self.budget <= 0:
            return 0.0
        return min(1.0, self.tokens_spent / self.budget)

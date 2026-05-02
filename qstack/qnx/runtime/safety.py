from __future__ import annotations

from typing import Any, Callable


class SafetyConstraints:
    def __init__(self, rules: list[Callable[[Any, Any], bool]]):
        self._rules = rules

    def evaluate(self, state: Any, goal: Any) -> bool:
        return all(rule(state, goal) for rule in self._rules)


class SafetyEnvelope:
    def __init__(self, bounds: dict[str, tuple]):
        """Bounds format: variable -> (low, high)."""

        self._bounds = bounds

    def inside(self, state) -> bool:
        for key, (low, high) in self._bounds.items():
            value = state.data.get(key)
            if value is None:
                continue
            if not (low <= value <= high):
                return False
        return True


class RateLimiter:
    """Simple deterministic rate limiter using counters stored in state."""

    def __init__(self, limit_key: str, max_calls: int):
        self.limit_key = limit_key
        self.max_calls = max_calls

    def increment_and_check(self, state) -> bool:
        count = state.read(self.limit_key, 0) + 1
        state.update(self.limit_key, count)
        return count <= self.max_calls


class SafetyValidator:
    def __init__(
        self,
        constraints: SafetyConstraints,
        envelope: SafetyEnvelope,
        rate_limiter: RateLimiter | None = None,
    ):
        self._constraints = constraints
        self._envelope = envelope
        self._rate_limiter = rate_limiter

    def pre_check(self, state, goal) -> bool:
        constraints_ok = self._constraints.evaluate(state, goal)
        rate_ok = (
            True if self._rate_limiter is None else self._rate_limiter.increment_and_check(state)
        )
        return constraints_ok and rate_ok

    def post_check(self, state, trace) -> bool:
        return self._envelope.inside(state)

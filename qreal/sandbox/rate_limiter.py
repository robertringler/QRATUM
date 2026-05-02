"""Tick-based deterministic rate limiting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateLimiter:
    interval: int

    def allow(self, tick: int) -> bool:
        if self.interval <= 0:
            return True
        return tick % self.interval == 0

"""Deterministic logical clock."""

from __future__ import annotations


class LogicalClock:
    def __init__(self) -> None:
        self._tick = 0

    def advance(self, steps: int = 1) -> int:
        self._tick += steps
        return self._tick

    def now(self) -> int:
        return self._tick

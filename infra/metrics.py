"""Deterministic metric counters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counter:
    name: str
    value: int = 0

    def inc(self, amount: int = 1) -> int:
        self.value += amount
        return self.value

    def snapshot(self) -> dict[str, int]:
        return {self.name: self.value}


class MetricRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def snapshot(self) -> dict[str, int]:
        return {name: counter.value for name, counter in sorted(self._counters.items())}

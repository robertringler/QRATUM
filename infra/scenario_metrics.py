"""Deterministic counters for scenario metrics."""
from __future__ import annotations


class ScenarioMetrics:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)

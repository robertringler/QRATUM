"""Deterministic counters for scenario metrics."""

from __future__ import annotations

from typing import Dict


class ScenarioMetrics:
    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counters)

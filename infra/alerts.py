"""Threshold-based alerting on deterministic metrics."""

from __future__ import annotations

from typing import Dict, List, Tuple


class Alert:
    def __init__(self, name: str, threshold: int) -> None:
        self.name = name
        self.threshold = threshold

    def evaluate(self, metric_value: int) -> bool:
        return metric_value >= self.threshold


class AlertEngine:
    def __init__(self) -> None:
        self._alerts: Dict[str, Alert] = {}
        self._triggered: List[Tuple[str, int]] = []

    def register(self, name: str, threshold: int) -> Alert:
        alert = Alert(name, threshold)
        self._alerts[name] = alert
        return alert

    def evaluate(self, metrics: Dict[str, int]) -> List[Tuple[str, int]]:
        self._triggered.clear()
        for name, alert in sorted(self._alerts.items()):
            if alert.evaluate(metrics.get(name, 0)):
                self._triggered.append((name, metrics.get(name, 0)))
        return list(self._triggered)

    def triggered(self) -> List[Tuple[str, int]]:
        return list(self._triggered)

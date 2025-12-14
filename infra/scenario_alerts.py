"""Threshold-based alerting for scenarios."""
from __future__ import annotations



def evaluate_alerts(metrics: dict[str, int], thresholds: dict[str, int]) -> list[str]:
    alerts: list[str] = []
    for name, limit in thresholds.items():
        value = metrics.get(name, 0)
        if value >= limit:
            alerts.append(f"alert:{name}={value}")
    return alerts

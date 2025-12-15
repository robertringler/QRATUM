"""Threshold-based alerting for scenarios."""

from __future__ import annotations

from typing import Dict, List


def evaluate_alerts(metrics: Dict[str, int], thresholds: Dict[str, int]) -> List[str]:
    alerts: List[str] = []
    for name, limit in thresholds.items():
        value = metrics.get(name, 0)
        if value >= limit:
            alerts.append(f"alert:{name}={value}")
    return alerts

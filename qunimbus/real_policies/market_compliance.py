"""Deterministic market compliance rules."""
from __future__ import annotations


def check_liquidity(snapshot: dict[str, float], minimum_volume: float) -> list[str]:
    violations: list[str] = []
    volume = float(snapshot.get("volume", 0.0))
    if volume < minimum_volume:
        violations.append("liquidity_breach")
    return violations

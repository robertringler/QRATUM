"""Deterministic market compliance rules."""

from __future__ import annotations

from typing import Dict, List


def check_liquidity(snapshot: Dict[str, float], minimum_volume: float) -> List[str]:
    violations: List[str] = []
    volume = float(snapshot.get("volume", 0.0))
    if volume < minimum_volume:
        violations.append("liquidity_breach")
    return violations

"""Aerospace flight envelope rules."""

from __future__ import annotations

from typing import Dict, List


def envelope_check(snapshot: Dict[str, float], max_velocity: float, max_angle: float) -> List[str]:
    violations: List[str] = []
    if float(snapshot.get("velocity", 0.0)) > max_velocity:
        violations.append("velocity_exceeded")
    if abs(float(snapshot.get("angle", 0.0))) > max_angle:
        violations.append("angle_exceeded")
    return violations

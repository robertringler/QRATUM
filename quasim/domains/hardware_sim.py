"""Simple deterministic hardware behavior simulation."""

from __future__ import annotations

from typing import Dict


def simulate_load(profile: Dict[str, int], duration: int) -> Dict[str, int]:
    return {k: v * duration for k, v in profile.items()}


def failure_probability(profile: Dict[str, int]) -> float:
    total = sum(profile.values())
    return min(1.0, total / 1000.0)

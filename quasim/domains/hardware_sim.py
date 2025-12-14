"""Simple deterministic hardware behavior simulation."""
from __future__ import annotations



def simulate_load(profile: dict[str, int], duration: int) -> dict[str, int]:
    return {k: v * duration for k, v in profile.items()}


def failure_probability(profile: dict[str, int]) -> float:
    total = sum(profile.values())
    return min(1.0, total / 1000.0)

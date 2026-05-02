"""Classical deterministic simulators."""

from __future__ import annotations

from typing import Callable


def linear_simulation(seed: float, steps: list[Callable[[float], float]]) -> list[float]:
    state = seed
    outputs = []
    for step in steps:
        state = step(state)
        outputs.append(state)
    return outputs

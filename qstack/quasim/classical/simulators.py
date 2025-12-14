"""Classical deterministic simulators."""

from __future__ import annotations

from typing import Callable, List


def linear_simulation(seed: float, steps: List[Callable[[float], float]]) -> List[float]:
    state = seed
    outputs = []
    for step in steps:
        state = step(state)
        outputs.append(state)
    return outputs

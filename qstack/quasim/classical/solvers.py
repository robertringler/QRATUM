"""Deterministic solvers."""
from __future__ import annotations



def fixed_point_solver(initial: float, coefficient: float, iterations: int = 4) -> float:
    state = initial
    for _ in range(iterations):
        state = coefficient * state + (1 - coefficient) * initial
    return state

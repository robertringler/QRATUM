"""Field evolution utilities for Quantacosmic simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .coupling_ops import CouplingOperator


@dataclass
class FieldLattice:
    """Discrete lattice used to evolve quantacosmic fields."""

    positions: Sequence[float]
    timestep: float

    def __post_init__(self) -> None:
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive.")
        if len(self.positions) < 1:
            raise ValueError("A lattice requires at least one position.")

    def propagation_kernel(self, coupling: CouplingOperator) -> list[list[float]]:
        base_matrix = coupling.matrix()
        if len(base_matrix) != len(self.positions):
            raise ValueError("Coupling matrix dimension must match lattice size.")
        scaled = [[element * self.timestep for element in row] for row in base_matrix]
        return scaled


def propagate_field(
    initial_field: Sequence[float],
    coupling: CouplingOperator,
    timestep: float,
    steps: int,
    nonlinearity: Callable[[float], float] | None = None,
) -> list[float]:
    """Propagate a field using a simple linear iterative scheme."""

    if steps < 0:
        raise ValueError("Steps must be non-negative.")
    if timestep <= 0:
        raise ValueError("Timestep must be positive.")

    lattice = FieldLattice(range(len(initial_field)), timestep)
    kernel = lattice.propagation_kernel(coupling)
    if any(len(row) != len(initial_field) for row in kernel):
        raise ValueError("Coupling matrix must be square and match field dimension.")

    state = [float(value) for value in initial_field]
    transform = nonlinearity or (lambda value: value)

    for _ in range(steps):
        updated = []
        for row in kernel:
            total = sum(weight * value for weight, value in zip(row, state))
            updated.append(transform(total))
        state = updated
    return state


__all__ = ["FieldLattice", "propagate_field"]

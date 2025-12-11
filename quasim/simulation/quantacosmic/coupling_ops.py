"""Coupling operators for Quantacosmic lattice interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CouplingOperator:
    """Simple dense coupling operator used by the quantacosmic lattice."""

    strengths: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        size = len(self.strengths)
        if size == 0:
            raise ValueError("Coupling operator requires at least one row.")
        for row in self.strengths:
            if len(row) != size:
                raise ValueError("Coupling operator must be square.")

    def matrix(self) -> list[list[float]]:
        return [list(map(float, row)) for row in self.strengths]

    def apply(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != len(self.strengths):
            raise ValueError("Vector dimensionality mismatch for coupling operator.")
        return [sum(weight * value for weight, value in zip(row, vector)) for row in self.strengths]


def coupling_matrix(strengths: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return a normalized coupling matrix enforcing symmetry."""

    operator = CouplingOperator(strengths)
    matrix = operator.matrix()

    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            average = (matrix[i][j] + matrix[j][i]) / 2.0
            matrix[i][j] = matrix[j][i] = average
    return matrix


__all__ = ["CouplingOperator", "coupling_matrix"]

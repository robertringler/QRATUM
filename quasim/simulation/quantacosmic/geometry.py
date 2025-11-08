"""Foundational geometric constructs for the Quantacosmic simulation space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def _validate_square(matrix: Sequence[Sequence[float]]) -> None:
    size = len(matrix)
    if size == 0:
        raise ValueError("Metric tensor must not be empty.")
    for row in matrix:
        if len(row) != size:
            raise ValueError("Metric tensor must be square.")


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    size = len(matrix)
    if size == 1:
        return float(matrix[0][0])
    if size == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])

    det = 0.0
    for column, value in enumerate(matrix[0]):
        minor = [
            [matrix[row][col] for col in range(len(matrix)) if col != column]
            for row in range(1, len(matrix))
        ]
        cofactor = ((-1) ** column) * value * _determinant(minor)
        det += cofactor
    return float(det)


@dataclass
class QuantumManifold:
    """Representation of the quantacosmic configuration space."""

    dimension: int
    metric: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        _validate_square(self.metric)
        if self.dimension != len(self.metric):
            raise ValueError("Dimension and metric size must agree.")

    @property
    def metric_tensor(self) -> Sequence[Sequence[float]]:
        return self.metric

    def volume_element(self) -> float:
        """Return the infinitesimal volume element derived from the metric determinant."""

        det = _determinant(self.metric)
        if det < 0:
            raise ValueError("Quantacosmic metric must yield a non-negative determinant.")
        return det ** 0.5

    def project_vector(self, vector: Sequence[float]) -> List[float]:
        """Project a vector into the manifold using the metric tensor."""

        if len(vector) != self.dimension:
            raise ValueError("Vector dimensionality mismatch.")
        projected = []
        for row in self.metric:
            projected.append(sum(component * value for component, value in zip(vector, row)))
        return projected


def curvature_scalar(metric: Sequence[Sequence[float]]) -> float:
    """Compute a lightweight scalar curvature proxy.

    The routine intentionally keeps the mathematics simple while providing a deterministic
    and well-behaved scalar that higher level components can exercise in tests.
    """

    _validate_square(metric)
    trace = sum(metric[i][i] for i in range(len(metric)))
    average_off_diagonal = 0.0
    off_diagonal_count = len(metric) ** 2 - len(metric)
    if off_diagonal_count:
        average_off_diagonal = sum(
            metric[row][col]
            for row in range(len(metric))
            for col in range(len(metric))
            if row != col
        ) / off_diagonal_count
    return trace - average_off_diagonal


__all__ = ["QuantumManifold", "curvature_scalar"]

"""Metric utilities for the Quantacosmic simulation suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import _determinant, _validate_square


@dataclass(frozen=True)
class MetricTensor:
    """Container for a symmetric metric tensor."""

    components: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        _validate_square(self.components)

    @property
    def dimension(self) -> int:
        return len(self.components)

    def trace(self) -> float:
        return sum(self.components[i][i] for i in range(self.dimension))

    def determinant(self) -> float:
        return _determinant(self.components)

    def normalize(self) -> "MetricTensor":
        det = self.determinant()
        if det == 0:
            raise ValueError("Cannot normalise a singular metric tensor.")
        factor = abs(det) ** (-1.0 / self.dimension)
        normalized = [
            [component * factor for component in row]
            for row in self.components
        ]
        return MetricTensor(normalized)


def revultra_temporal_curvature(
    metric: MetricTensor,
    temporal_frequency: float,
    cognitive_twist: float = 1.0,
) -> float:
    """Return a REVULTRA curvature quotient derived from the metric trace.

    The quantity is intentionally simple yet non-trivial; higher temporal frequencies and
    cognitive twist factors scale the contribution from the normalized trace.
    """

    if temporal_frequency < 0:
        raise ValueError("Temporal frequency must be non-negative.")
    normalized_metric = metric.normalize()
    trace = normalized_metric.trace()
    return temporal_frequency * cognitive_twist * trace / normalized_metric.dimension


__all__ = ["MetricTensor", "revultra_temporal_curvature"]

"""Analog gravity and metamaterial simulation modules."""

from .metric import (
    acoustic_black_hole_metric,
    check_normalization_preservation,
    effective_potential_from_metric,
    geodesic_propagation,
    refractive_index_to_metric,
)

__all__ = [
    "refractive_index_to_metric",
    "geodesic_propagation",
    "check_normalization_preservation",
    "effective_potential_from_metric",
    "acoustic_black_hole_metric",
]

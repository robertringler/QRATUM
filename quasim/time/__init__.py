"""Temporal dynamics and correlation modules."""

from .interference import (compute_emission_intensity, interference_pattern,
                           multi_state_superposition_intensity,
                           rabi_oscillation)

__all__ = [
    "compute_emission_intensity",
    "rabi_oscillation",
    "interference_pattern",
    "multi_state_superposition_intensity",
]

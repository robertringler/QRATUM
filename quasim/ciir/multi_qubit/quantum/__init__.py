"""Quantum Simulation Agent — density-matrix Lindblad solver."""

from .density_matrix import (
    DensityMatrixSimulator,
    LindbladParams,
    amplitude_damping_ops,
    dephasing_ops,
    depolarizing_ops,
    evolve_rk4,
    lindblad_superop,
)

__all__ = [
    "DensityMatrixSimulator",
    "LindbladParams",
    "amplitude_damping_ops",
    "dephasing_ops",
    "depolarizing_ops",
    "evolve_rk4",
    "lindblad_superop",
]

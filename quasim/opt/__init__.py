"""Optimization module for QuASIM.

This module provides quantum-enhanced optimization algorithms
including quantum annealing, QAOA, and hybrid classical-quantum solvers,
as well as large-scale graph algorithms like UltraSSSP.
"""

from __future__ import annotations

from .classical_fallback import ClassicalFallback
from .graph import HierarchicalGraph, QGraph
from .optimizer import QuantumOptimizer
from .problems import OptimizationProblem
from .ultra_sssp import (
    SSSPSimulationConfig,
    UltraSSSP,
    dijkstra_baseline,
    run_sssp_simulation,
    validate_sssp_results,
)

__all__ = [
    "QuantumOptimizer",
    "OptimizationProblem",
    "ClassicalFallback",
    "QGraph",
    "HierarchicalGraph",
    "UltraSSSP",
    "dijkstra_baseline",
    "validate_sssp_results",
    "SSSPSimulationConfig",
    "run_sssp_simulation",
]

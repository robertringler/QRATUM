"""Optimization module for QuASIM.

This module provides quantum-enhanced optimization algorithms
including quantum annealing, QAOA, and hybrid classical-quantum solvers.
"""

from __future__ import annotations

from .classical_fallback import ClassicalFallback
from .optimizer import QuantumOptimizer
from .problems import OptimizationProblem

__all__ = ["QuantumOptimizer", "OptimizationProblem", "ClassicalFallback"]

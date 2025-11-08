"""Optimization problem definitions."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OptimizationProblem:
    """Base class for optimization problems.

    Defines the interface for optimization problems that can be solved
    using QuASIM's quantum-enhanced algorithms.

    Attributes:
        name: Problem name/identifier
        dimension: Problem dimension (number of variables)
        is_minimization: Whether this is a minimization problem
        bounds: Variable bounds as list of (min, max) tuples
        constraints: Problem constraints (optional)
    """

    name: str
    dimension: int
    is_minimization: bool = True
    bounds: list[tuple[float, float]] = field(default_factory=list)
    constraints: list[Callable] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default bounds if not provided."""
        if not self.bounds:
            self.bounds = [(0.0, 1.0) for _ in range(self.dimension)]

    def evaluate(self, solution: list[float]) -> float:
        """Evaluate the objective function at a given solution.

        Args:
            solution: Candidate solution (list of variable values)

        Returns:
            Objective function value
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def get_random_solution(self) -> list[float]:
        """Generate a random feasible solution.

        Returns:
            Random solution within variable bounds
        """
        solution = []
        for lower, upper in self.bounds:
            value = random.uniform(lower, upper)
            solution.append(value)
        return solution

    def is_feasible(self, solution: list[float]) -> bool:
        """Check if a solution satisfies all constraints.

        Args:
            solution: Candidate solution

        Returns:
            True if solution is feasible
        """
        # Check bounds
        for i, (lower, upper) in enumerate(self.bounds):
            if solution[i] < lower or solution[i] > upper:
                return False

        # Check additional constraints
        for constraint in self.constraints:
            if not constraint(solution):
                return False

        return True


@dataclass
class PortfolioOptimization(OptimizationProblem):
    """Portfolio optimization problem for finance applications.

    Optimize asset allocation to maximize returns while minimizing risk.
    This problem is ideal for quantum optimization due to its combinatorial
    nature and the quadratic objective function.
    """

    expected_returns: list[float] = field(default_factory=list)
    covariance_matrix: list[list[float]] = field(default_factory=list)
    risk_aversion: float = 1.0

    def evaluate(self, solution: list[float]) -> float:
        """Evaluate portfolio performance.

        Uses mean-variance optimization framework.

        Args:
            solution: Portfolio weights (must sum to 1)

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        # Simplified portfolio evaluation
        expected_return = sum(w * r for w, r in zip(solution, self.expected_returns))

        # Calculate portfolio variance (simplified)
        variance = sum(solution[i] ** 2 for i in range(len(solution)))

        # Return negative Sharpe ratio (for minimization)
        if variance == 0:
            return float("inf")

        sharpe_ratio = expected_return / (variance**0.5)
        return -sharpe_ratio  # Negative because we minimize


@dataclass
class MolecularOptimization(OptimizationProblem):
    """Molecular structure optimization for pharmaceutical applications.

    Optimize molecular structure to minimize energy or maximize binding affinity.
    Leverages quantum computing for accurate electronic structure calculations.
    """

    target_properties: dict[str, float] = field(default_factory=dict)

    def evaluate(self, solution: list[float]) -> float:
        """Evaluate molecular energy or binding affinity.

        Args:
            solution: Molecular configuration parameters

        Returns:
            Energy or negative binding affinity
        """
        # Simplified molecular energy calculation
        # Production version would use quantum chemistry methods (VQE)
        energy = sum(x**2 for x in solution)
        return energy

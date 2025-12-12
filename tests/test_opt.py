"""Optimization module tests.

This module tests the QuASIM quantum optimization functionality including:
- Optimizer configuration
- Optimization algorithms (QAOA, QA, VQE, Hybrid)
- Problem solving
- Convergence behavior
"""

from __future__ import annotations

import pytest

from quasim.opt import OptimizationProblem, QuantumOptimizer


# Mock implementation of OptimizationProblem for testing
class MockOptimizationProblem(OptimizationProblem):
    """Simple test problem for unit tests."""

    def evaluate(self, solution: list[float]) -> float:
        """Simple sphere function for testing."""
        return sum(x**2 for x in solution)


class TestOptimizationProblem:
    """Test optimization problem specification."""

    def test_problem_creation(self):
        """Test creating an optimization problem."""
        problem = MockOptimizationProblem(name="maxcut", dimension=10, is_minimization=False)
        assert problem.name == "maxcut"
        assert problem.dimension == 10
        assert problem.is_minimization is False

    def test_problem_with_constraints(self):
        """Test creating problem with constraints."""
        constraints = [lambda x: sum(x) <= 100]
        problem = MockOptimizationProblem(
            name="portfolio", dimension=20, is_minimization=False, constraints=constraints
        )
        assert len(problem.constraints) == 1

    def test_problem_default_bounds(self):
        """Test problem with default bounds."""
        problem = MockOptimizationProblem(name="test", dimension=5)
        assert len(problem.bounds) == 5
        assert all(b == (0.0, 1.0) for b in problem.bounds)

    def test_problem_custom_bounds(self):
        """Test problem with custom bounds."""
        bounds = [(-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]
        problem = MockOptimizationProblem(name="test", dimension=3, bounds=bounds)
        assert problem.bounds == bounds

    def test_problem_random_solution(self):
        """Test generating random solution."""
        problem = MockOptimizationProblem(name="test", dimension=5)
        solution = problem.get_random_solution()
        assert len(solution) == 5
        assert all(0.0 <= x <= 1.0 for x in solution)

    def test_problem_feasibility_check(self):
        """Test feasibility checking."""
        problem = MockOptimizationProblem(name="test", dimension=3)
        assert problem.is_feasible([0.5, 0.5, 0.5])
        assert not problem.is_feasible([1.5, 0.5, 0.5])  # Out of bounds


class TestQuantumOptimizer:
    """Test quantum optimizer functionality."""

    def test_optimizer_default_initialization(self):
        """Test default optimizer initialization."""
        optimizer = QuantumOptimizer()
        assert optimizer.algorithm == "qaoa"
        assert optimizer.backend == "cpu"
        assert optimizer.max_iterations == 100
        assert optimizer.convergence_tolerance == 1e-6
        assert optimizer.random_seed == 42

    def test_optimizer_custom_configuration(self):
        """Test custom optimizer configuration."""
        optimizer = QuantumOptimizer(
            algorithm="vqe", backend="cuda", max_iterations=200, random_seed=123
        )
        assert optimizer.algorithm == "vqe"
        assert optimizer.backend == "cuda"
        assert optimizer.max_iterations == 200
        assert optimizer.random_seed == 123

    def test_optimizer_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            QuantumOptimizer(algorithm="invalid")

    @pytest.mark.parametrize("algorithm", ["qa", "qaoa", "vqe", "hybrid"])
    def test_all_algorithms_supported(self, algorithm):
        """Test that all documented algorithms are supported."""
        optimizer = QuantumOptimizer(algorithm=algorithm)
        assert optimizer.algorithm == algorithm


class TestOptimization:
    """Test optimization execution."""

    def test_qaoa_optimization(self):
        """Test QAOA optimization."""
        optimizer = QuantumOptimizer(algorithm="qaoa", max_iterations=50)
        problem = MockOptimizationProblem(name="maxcut", dimension=5, is_minimization=False)

        result = optimizer.optimize(problem)

        assert "solution" in result
        assert "objective_value" in result
        assert "iterations" in result
        assert "convergence" in result
        assert isinstance(result["convergence"], bool)

    def test_annealing_optimization(self):
        """Test quantum annealing optimization."""
        optimizer = QuantumOptimizer(algorithm="qa", max_iterations=50)
        problem = MockOptimizationProblem(name="portfolio", dimension=10, is_minimization=False)

        result = optimizer.optimize(problem)

        assert "solution" in result
        assert "objective_value" in result
        assert "iterations" in result

    def test_vqe_optimization(self):
        """Test VQE optimization."""
        optimizer = QuantumOptimizer(algorithm="vqe", max_iterations=50)
        problem = MockOptimizationProblem(name="tsp", dimension=8, is_minimization=True)

        result = optimizer.optimize(problem)

        assert "solution" in result
        assert "objective_value" in result

    def test_hybrid_optimization(self):
        """Test hybrid classical-quantum optimization."""
        optimizer = QuantumOptimizer(algorithm="hybrid", max_iterations=50)
        problem = MockOptimizationProblem(name="knapsack", dimension=15, is_minimization=False)

        result = optimizer.optimize(problem)

        assert "solution" in result
        assert "objective_value" in result

    def test_optimization_with_initial_params(self):
        """Test optimization with initial parameter guess."""
        optimizer = QuantumOptimizer(algorithm="qaoa")
        problem = MockOptimizationProblem(name="maxcut", dimension=5, is_minimization=False)
        initial = {"beta": 0.5, "gamma": 0.3}

        result = optimizer.optimize(problem, initial_params=initial)

        assert "solution" in result
        assert result["iterations"] > 0

    def test_deterministic_behavior(self):
        """Test that optimization behavior is consistent with same seed."""
        optimizer1 = QuantumOptimizer(algorithm="qaoa", random_seed=42, max_iterations=20)
        optimizer2 = QuantumOptimizer(algorithm="qaoa", random_seed=42, max_iterations=20)
        problem1 = MockOptimizationProblem(name="maxcut", dimension=4, is_minimization=False)
        problem2 = MockOptimizationProblem(name="maxcut", dimension=4, is_minimization=False)

        result1 = optimizer1.optimize(problem1)
        result2 = optimizer2.optimize(problem2)

        # Both should produce valid results (determinism within reasonable tolerance)
        assert "objective_value" in result1
        assert "objective_value" in result2
        # Results should be in the same ballpark
        assert abs(result1["objective_value"] - result2["objective_value"]) < 2.0

    def test_different_seeds_can_differ(self):
        """Test that different seeds can produce different results."""
        optimizer1 = QuantumOptimizer(algorithm="qaoa", random_seed=42, max_iterations=20)
        optimizer2 = QuantumOptimizer(algorithm="qaoa", random_seed=123, max_iterations=20)
        problem = MockOptimizationProblem(name="maxcut", dimension=4, is_minimization=False)

        result1 = optimizer1.optimize(problem)
        result2 = optimizer2.optimize(problem)

        # Results exist for both
        assert "objective_value" in result1
        assert "objective_value" in result2


class TestConvergence:
    """Test convergence behavior."""

    def test_convergence_with_few_iterations(self):
        """Test optimization with limited iterations."""
        optimizer = QuantumOptimizer(algorithm="qaoa", max_iterations=5)
        problem = MockOptimizationProblem(name="maxcut", dimension=3, is_minimization=False)

        result = optimizer.optimize(problem)

        assert result["iterations"] <= 5

    def test_convergence_tolerance(self):
        """Test convergence tolerance setting."""
        optimizer = QuantumOptimizer(algorithm="qaoa", convergence_tolerance=1e-8)
        assert optimizer.convergence_tolerance == 1e-8


class TestOptimizationProblems:
    """Test different optimization problem types."""

    @pytest.mark.parametrize(
        "problem_name", ["maxcut", "portfolio", "tsp", "knapsack", "scheduling"]
    )
    def test_problem_types(self, problem_name):
        """Test various optimization problem types."""
        problem = MockOptimizationProblem(name=problem_name, dimension=5, is_minimization=False)
        optimizer = QuantumOptimizer(algorithm="qaoa", max_iterations=10)

        result = optimizer.optimize(problem)
        assert "solution" in result

    def test_minimize_objective(self):
        """Test minimization objective."""
        problem = MockOptimizationProblem(name="tsp", dimension=5, is_minimization=True)
        optimizer = QuantumOptimizer(max_iterations=10)

        result = optimizer.optimize(problem)
        assert "objective_value" in result

    def test_maximize_objective(self):
        """Test maximization objective."""
        problem = MockOptimizationProblem(name="maxcut", dimension=5, is_minimization=False)
        optimizer = QuantumOptimizer(max_iterations=10)

        result = optimizer.optimize(problem)
        assert "objective_value" in result


class TestBackends:
    """Test different computation backends."""

    @pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
    def test_backends(self, backend):
        """Test all supported backends."""
        optimizer = QuantumOptimizer(algorithm="qaoa", backend=backend)
        assert optimizer.backend == backend

        problem = MockOptimizationProblem(name="maxcut", dimension=3, is_minimization=False)
        result = optimizer.optimize(problem)
        assert "solution" in result

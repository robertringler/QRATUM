"""Tests for classical optimization fallback module."""

import pytest

from quasim.opt.classical_fallback import ClassicalFallback


class TestClassicalFallbackInit:
    """Test ClassicalFallback initialization."""

    def test_default_init(self):
        """Test default initialization."""
        fallback = ClassicalFallback()
        assert fallback.max_problem_size == 20

    def test_custom_max_problem_size(self):
        """Test custom max problem size."""
        fallback = ClassicalFallback(max_problem_size=10)
        assert fallback.max_problem_size == 10


class TestMaxCutClassical:
    """Test classical MaxCut solvers."""

    def test_solve_maxcut_exact_small(self):
        """Test exact MaxCut solver on small graph."""
        fallback = ClassicalFallback(max_problem_size=10)
        edges = [(0, 1), (1, 2), (2, 0)]

        result = fallback.solve_maxcut(edges, method="exact")

        assert result["classical"] is True
        assert result["optimal"] is True
        assert result["cut_value"] >= 2  # Triangle has max cut of 2
        assert len(result["solution"]) == 3

    def test_solve_maxcut_exact_too_large(self):
        """Test that exact solver raises error for large graphs."""
        fallback = ClassicalFallback(max_problem_size=5)
        # Create graph with 6 nodes
        edges = [(i, i + 1) for i in range(5)] + [(5, 0)]

        with pytest.raises(ValueError, match="Problem too large"):
            fallback.solve_maxcut(edges, method="exact")

    def test_solve_maxcut_greedy(self):
        """Test greedy MaxCut solver."""
        fallback = ClassicalFallback()
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        result = fallback.solve_maxcut(edges, method="greedy")

        assert result["classical"] is True
        assert result["optimal"] is False  # Greedy is heuristic
        assert result["cut_value"] > 0
        assert len(result["solution"]) == 4

    def test_solve_maxcut_unknown_method(self):
        """Test that unknown method raises ValueError."""
        fallback = ClassicalFallback()
        edges = [(0, 1)]

        with pytest.raises(ValueError, match="Unknown method"):
            fallback.solve_maxcut(edges, method="invalid")

"""Enhanced validation tests for QuASIM kernels.

This module provides:
- Property-based tests using Hypothesis
- Determinism tests (reproducibility with fixed seeds)
- Numerical precision tests
- Edge case handling
"""
from __future__ import annotations

import random
from typing import List

import pytest
from hypothesis import given, strategies as st, settings

from quasim.runtime import Config, runtime
from sdk.rapids.dataframe import columnar_sum


# Property-based tests for K001: Tensor Contraction
@given(
    batches=st.integers(min_value=1, max_value=10),
    dimension=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20, deadline=5000)
def test_k001_property_output_length(batches: int, dimension: int):
    """Property: Output length should equal number of batches."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [
        [complex(i, -i) for i in range(dimension)]
        for _ in range(batches)
    ]
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    assert len(result) == batches, f"Expected {batches} results, got {len(result)}"


@given(
    dimension=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=10, deadline=5000)
def test_k001_property_zero_input(dimension: int):
    """Property: All-zero input should produce zero output."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [[complex(0, 0) for _ in range(dimension)]]
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    assert len(result) == 1
    assert abs(result[0]) < 1e-10, f"Expected ~0, got {result[0]}"


@given(
    value=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    dimension=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=10, deadline=5000)
def test_k001_property_constant_input(value: float, dimension: int):
    """Property: Constant input should produce predictable sum."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [[complex(value, 0) for _ in range(dimension)]]
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    expected = complex(value * dimension, 0)
    # Allow some tolerance for floating-point arithmetic
    assert abs(result[0] - expected) < abs(expected) * 1e-6 + 1e-10, \
        f"Expected {expected}, got {result[0]}"


# Determinism tests for K001
def test_k001_determinism():
    """Verify that K001 produces identical results across runs."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    
    # Generate deterministic circuit
    random.seed(42)
    circuit = [
        [complex(random.random(), random.random()) for _ in range(100)]
        for _ in range(5)
    ]
    
    # Run multiple times
    results = []
    for _ in range(3):
        with runtime(cfg) as rt:
            result = rt.simulate(circuit)
        results.append(result)
    
    # All results should be identical
    for i in range(1, len(results)):
        for j in range(len(results[0])):
            assert results[0][j] == results[i][j], \
                f"Non-deterministic result at run {i}, element {j}"


# Property-based tests for K003: Columnar Sum
@given(
    num_columns=st.integers(min_value=1, max_value=20),
    num_rows=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20, deadline=5000)
def test_k003_property_output_columns(num_columns: int, num_rows: int):
    """Property: Output should have same number of columns as input."""
    table = {
        f"col_{i}": [float(j) for j in range(num_rows)]
        for i in range(num_columns)
    }
    
    result = columnar_sum(table)
    
    assert len(result) == num_columns
    assert set(result.keys()) == set(table.keys())


@given(
    num_columns=st.integers(min_value=1, max_value=10),
    num_rows=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=10, deadline=5000)
def test_k003_property_zero_table(num_columns: int, num_rows: int):
    """Property: All-zero table should produce all-zero sums."""
    table = {
        f"col_{i}": [0.0] * num_rows
        for i in range(num_columns)
    }
    
    result = columnar_sum(table)
    
    for col_name, sum_value in result.items():
        assert abs(sum_value) < 1e-10, f"Expected ~0 for {col_name}, got {sum_value}"


@given(
    value=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    num_rows=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=10, deadline=5000)
def test_k003_property_constant_column(value: float, num_rows: int):
    """Property: Constant column should sum to value * num_rows."""
    table = {"test_col": [value] * num_rows}
    
    result = columnar_sum(table)
    
    expected = value * num_rows
    assert abs(result["test_col"] - expected) < abs(expected) * 1e-6 + 1e-10, \
        f"Expected {expected}, got {result['test_col']}"


# Determinism tests for K003
def test_k003_determinism():
    """Verify that K003 produces identical results across runs."""
    random.seed(42)
    table = {
        f"col_{i}": [random.uniform(-100, 100) for _ in range(100)]
        for i in range(10)
    }
    
    # Run multiple times
    results = []
    for _ in range(3):
        results.append(columnar_sum(table))
    
    # All results should be identical
    for i in range(1, len(results)):
        for col_name in results[0].keys():
            assert results[0][col_name] == results[i][col_name], \
                f"Non-deterministic result for {col_name} at run {i}"


# Edge case tests
def test_k001_empty_tensor():
    """Test K001 with empty tensor (dimension=0)."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [[]]  # One batch with no elements
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    assert len(result) == 1
    assert result[0] == complex(0, 0)


def test_k001_single_element():
    """Test K001 with single-element tensors."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [[complex(42, -42)] for _ in range(3)]
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    assert len(result) == 3
    for value in result:
        assert value == complex(42, -42)


def test_k003_empty_table():
    """Test K003 with empty table (no columns)."""
    table = {}
    result = columnar_sum(table)
    assert result == {}


def test_k003_empty_columns():
    """Test K003 with empty columns (no rows)."""
    table = {"col_a": [], "col_b": []}
    result = columnar_sum(table)
    assert result == {"col_a": 0.0, "col_b": 0.0}


# Numerical precision tests
def test_k001_large_magnitude_values():
    """Test K001 with large magnitude complex values."""
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    large_val = 1e6
    circuit = [[complex(large_val, -large_val) for _ in range(10)]]
    
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
    
    expected = complex(large_val * 10, -large_val * 10)
    relative_error = abs(result[0] - expected) / abs(expected)
    assert relative_error < 1e-6, f"Large magnitude precision error: {relative_error}"


def test_k003_large_magnitude_sums():
    """Test K003 with large magnitude values."""
    large_val = 1e8
    table = {"big_col": [large_val] * 100}
    
    result = columnar_sum(table)
    
    expected = large_val * 100
    relative_error = abs(result["big_col"] - expected) / expected
    assert relative_error < 1e-6, f"Large magnitude precision error: {relative_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

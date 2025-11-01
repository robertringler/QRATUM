"""Tests for autonomous systems kernel."""
import pytest
from quasim_fallback import autonomous_systems_kernel

def test_kernel_deterministic():
    """Test that kernel returns deterministic results for same seed."""
    result1 = autonomous_systems_kernel(seed=0, scale=1.0)
    result2 = autonomous_systems_kernel(seed=0, scale=1.0)
    assert result1["state_vector"] == result2["state_vector"]

def test_kernel_scale():
    """Test that scale parameter affects output."""
    result1 = autonomous_systems_kernel(seed=0, scale=1.0)
    result2 = autonomous_systems_kernel(seed=0, scale=2.0)
    assert result1["energy"] < result2["energy"]

def test_kernel_output_structure():
    """Test that kernel returns expected structure."""
    result = autonomous_systems_kernel(seed=0, scale=1.0)
    assert "state_vector" in result
    assert "energy" in result
    assert "convergence" in result
    assert len(result["state_vector"]) == 10

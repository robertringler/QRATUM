"""Smoke tests for finance demo."""

from quasim.demos.finance.kernels.simulation import run_simulation


def test_run_simulation_basic():
    """Test basic simulation."""

    scenario = {"default": True}
    results = run_simulation(scenario, steps=50, seed=42)

    assert "trace" in results
    assert "final_value" in results
    assert len(results["trace"]) == 50


def test_run_simulation_deterministic():
    """Test deterministic behavior."""

    scenario = {"default": True}

    results1 = run_simulation(scenario, steps=30, seed=42)
    results2 = run_simulation(scenario, steps=30, seed=42)

    assert abs(results1["final_value"] - results2["final_value"]) < 1e-6


def test_quick_run():
    """Quick integration test."""

    scenario = {"default": True}
    results = run_simulation(scenario, steps=20, seed=42)

    assert len(results["trace"]) == 20

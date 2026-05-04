"""Smoke tests for aerospace demo."""

from quasim.demos.aerospace.kernels.ascent import simulate_ascent
from quasim.demos.aerospace.scenarios.hot_staging import load_scenario


def test_simulate_ascent_basic():
    """Test basic ascent simulation."""

    scenario = {
        "mass_kg": 549000,
        "thrust_n": 7607000,
        "isp_s": 282,
    }

    results = simulate_ascent(scenario, steps=50, seed=42)

    assert "rmse_altitude" in results
    assert "rmse_velocity" in results
    assert "q_max" in results
    assert "fuel_margin" in results
    assert results["final_altitude"] > 0
    assert results["final_velocity"] > 0


def test_simulate_ascent_deterministic():
    """Test deterministic behavior."""

    scenario = load_scenario("falcon9")

    results1 = simulate_ascent(scenario, steps=30, seed=42)
    results2 = simulate_ascent(scenario, steps=30, seed=42)

    assert abs(results1["rmse_altitude"] - results2["rmse_altitude"]) < 1e-6
    assert abs(results1["rmse_velocity"] - results2["rmse_velocity"]) < 1e-6
    assert abs(results1["q_max"] - results2["q_max"]) < 1e-6


def test_load_scenarios():
    """Test scenario loading."""

    scenarios = ["starship", "falcon9", "hot_staging_v1"]

    for name in scenarios:
        scenario = load_scenario(name)
        assert "mass_kg" in scenario
        assert "thrust_n" in scenario
        assert "isp_s" in scenario


def test_quick_run():
    """Quick integration test."""

    scenario = load_scenario("starship")
    results = simulate_ascent(scenario, steps=20, seed=42)

    assert len(results["trace"]) == 20
    assert results["rmse_altitude"] > 0
    assert 0 <= results["fuel_margin"] <= 100

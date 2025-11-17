from qnx import QNXSubstrate, SimulationConfig


def test_qnx_substrate_initialises():
    substrate = QNXSubstrate()
    assert substrate.backends


def test_qnx_run_with_modern_backend():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="smoke", timesteps=2, seed=42)
    result = substrate.run_simulation(config)

    assert result.backend == "quasim_modern"
    assert result.simulation_hash
    assert "raw_results" in result.__dict__
    assert result.raw_results.get("engine") == "quasim_modern"

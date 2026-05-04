from qnx import QNXSubstrate, SimulationConfig


def test_quasim_modern_scenario_contract():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="deterministic", timesteps=3, seed=1234)

    result = substrate.run_simulation(config)

    assert result.backend == "quasim_modern"
    assert result.raw_results["scenario_id"] == "deterministic"
    assert result.raw_results["raw_output"]["result"]
    assert result.execution_time_ms >= 0


def test_modern_engine_is_deterministic_when_reused():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="repeatable", timesteps=1, seed=5)

    first = substrate.run_simulation(config)
    second = substrate.dispatch(config)

    assert first.simulation_hash == second.simulation_hash
    assert first.raw_results["raw_output"]["result"] == second.raw_results["raw_output"]["result"]

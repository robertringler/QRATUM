import pytest

from qnx import QNXSubstrate, SimulationConfig


def test_modern_backend_deterministic_hash():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="integration", timesteps=3, seed=11)

    first = substrate.run_simulation(config)
    second = substrate.run_simulation(config)

    assert first.simulation_hash == second.simulation_hash


def test_legacy_backend_stub_through_substrate():
    substrate = QNXSubstrate()
    config = SimulationConfig(
        scenario_id="legacy", timesteps=2, seed=1, backend="quasim_legacy_v1_2_0"
    )

    result = substrate.run_simulation(config)

    assert result.backend == "quasim_legacy_v1_2_0"
    assert result.raw_results["status"] == "not_implemented"


@pytest.mark.skipif(False, reason="Always execute")
def test_qvr_backend_unavailable_on_non_windows():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="smoke", timesteps=1, seed=0, backend="qvr_win")

    with pytest.raises(RuntimeError):
        substrate.run_simulation(config)

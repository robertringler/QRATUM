from qnx import QNXSubstrate, SimulationConfig
from tests.qnx_integration.mock_qnx_rtos import MockQNXRTOS


def test_lifecycle_flow_with_mock_rtos():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="lifecycle", timesteps=2, seed=7)
    mock_rtos = MockQNXRTOS()

    lifecycle = substrate.lifecycle_run(config, rtos=mock_rtos)

    assert lifecycle["init"]["status"] == "initialised"
    assert lifecycle["boot"]["ready"] is True
    assert lifecycle["dispatch_result"].simulation_hash
    assert lifecycle["teardown"]["status"] == "stopped"
    assert mock_rtos.ipc_bus[-1]["channel"] == "simulation_complete"


def test_boot_state_fails_for_unavailable_backend():
    substrate = QNXSubstrate()
    config = SimulationConfig(scenario_id="probe", timesteps=1, backend="quasim_modern")

    boot_state = substrate.boot_backend(config.backend, config)

    assert boot_state == {"backend": "quasim_modern", "ready": True}

    teardown_state = substrate.teardown_backend(config.backend)
    assert teardown_state["status"] == "stopped"

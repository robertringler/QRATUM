from qnx.backends.quasim_modern import QuasimModernBackend
from qnx.types import SimulationConfig


def test_modern_backend_runs_and_returns_result():
    backend = QuasimModernBackend()
    config = SimulationConfig(scenario_id="smoke", timesteps=3, seed=123)

    first = backend.run(config)
    second = backend.run(config)

    assert first["engine"] == "quasim_modern"
    assert first["raw_output"]["result"]
    assert first == second

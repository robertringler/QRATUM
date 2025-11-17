from qnx.backends.quasim_legacy_v120 import QuasimLegacyV120Backend
from qnx.types import SimulationConfig


def test_legacy_backend_stub():
    backend = QuasimLegacyV120Backend()
    config = SimulationConfig(scenario_id="legacy", timesteps=2, seed=1)

    result = backend.run(config)

    assert result["engine"] == "quasim_legacy_v1_2_0"
    assert result["status"] == "not_implemented"

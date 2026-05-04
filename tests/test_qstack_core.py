from __future__ import annotations

from qstack.config import QStackConfig
from qstack.qnx_adapter import QNXAdapter
from qstack.system import QStackSystem


def test_qstack_config_to_dict_keys():
    config = QStackConfig()
    config_dict = config.to_dict()
    assert set(config_dict.keys()) == {"qnx", "quasim", "qunimbus"}


def test_qnx_adapter_builds_simulation_config():
    config = QStackConfig()
    adapter = QNXAdapter(config.qnx)
    sim_config = adapter._build_sim_config()

    assert sim_config.scenario_id == config.qnx.scenario_id
    assert sim_config.timesteps == config.qnx.timesteps
    assert sim_config.seed == config.qnx.seed
    assert sim_config.backend == config.qnx.backend
    assert sim_config.security_level is not None
    assert sim_config.parameters is None


def test_qstack_system_initialises():
    config = QStackConfig()
    system = QStackSystem(config)

    assert system.config is config
    assert system.config.to_dict()["qnx"]["backend"] == "quasim_modern"

"""Shared fixtures for the FLUXA test suite."""

from __future__ import annotations

import pytest

from fluxa.config import DEFAULT_SCENARIO_CONFIG, load_system
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.network import DCNetwork
from fluxa.profiles import TimeGrid, build_series
from fluxa.scenarios import load_scenarios

#: Short horizon used by most tests. 48 steps at 5 min = 4 hours: long enough
#: to exercise ramping, storage and the receding horizon, short enough to keep
#: the suite fast.
SHORT_STEPS = 48


@pytest.fixture(scope="session")
def loaded():
    return load_system()


@pytest.fixture(scope="session")
def system(loaded):
    return loaded.system


@pytest.fixture(scope="session")
def network(system):
    return DCNetwork.from_system(system)


@pytest.fixture(scope="session")
def scenarios():
    return load_scenarios(DEFAULT_SCENARIO_CONFIG)


@pytest.fixture(scope="session")
def time_grid():
    return TimeGrid("2026-01-05T00:00:00Z", 300.0, SHORT_STEPS)


@pytest.fixture(scope="session")
def series(system, time_grid):
    return build_series(system, time_grid)


def make_engine(loaded, *, run_id="test", n_steps=SHORT_STEPS, **kwargs):
    """Build an engine with test defaults."""
    config = RunConfig(
        run_id=run_id,
        n_steps=n_steps,
        checkpoint_every=kwargs.pop("checkpoint_every", 12),
        authorized=kwargs.pop("authorized", True),
        authorizer_id=kwargs.pop("authorizer_id", "test_operator"),
        **kwargs,
    )
    return SimulationEngine(loaded, config)


@pytest.fixture(scope="session")
def baseline_result(loaded, scenarios):
    """A single short baseline run, shared across read-only tests."""
    return make_engine(loaded, run_id="baseline-fixture").run(scenarios["A_BASELINE"])

"""Tests for QCMG field simulation module."""

from __future__ import annotations

import pytest

from quasim.sim import (FieldState, QCMGParameters,
                        QuantacosmorphysigeneticField)


def test_qcmg_parameters_default():
    """Test QCMGParameters with default values."""

    params = QCMGParameters()
    assert params.grid_size == 64
    assert params.dt == 0.01
    assert params.coupling_strength == 0.1
    assert params.interaction_strength == 0.05
    assert params.thermal_noise == 0.001
    assert params.random_seed is None


def test_qcmg_parameters_custom():
    """Test QCMGParameters with custom values."""

    params = QCMGParameters(
        grid_size=32,
        dt=0.02,
        coupling_strength=0.2,
        interaction_strength=0.1,
        thermal_noise=0.002,
        random_seed=123,
    )
    assert params.grid_size == 32
    assert params.dt == 0.02
    assert params.coupling_strength == 0.2
    assert params.interaction_strength == 0.1
    assert params.thermal_noise == 0.002
    assert params.random_seed == 123


def test_field_state():
    """Test FieldState dataclass."""

    state = FieldState(time=1.0, coherence=0.5, entropy=2.0, energy=100.0)
    assert state.time == 1.0
    assert state.coherence == 0.5
    assert state.entropy == 2.0
    assert state.energy == 100.0


def test_field_initialization_gaussian():
    """Test field initialization with Gaussian mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    # Check that initial state was recorded
    state = field.get_state()
    assert state.time == 0.0
    assert 0.0 <= state.coherence <= 1.0
    assert state.entropy > 0.0
    assert state.energy > 0.0

    # Check history
    history = field.get_history()
    assert len(history) == 1


def test_field_initialization_uniform():
    """Test field initialization with uniform mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="uniform")

    state = field.get_state()
    assert state.time == 0.0
    assert 0.0 <= state.coherence <= 1.0


def test_field_initialization_zero():
    """Test field initialization with zero mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="zero")

    state = field.get_state()
    assert state.time == 0.0


def test_field_initialization_invalid_mode():
    """Test field initialization with invalid mode raises error."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(ValueError, match="Unknown initialization mode"):
        field.initialize(mode="invalid")


def test_field_evolution():
    """Test field evolution over multiple steps."""

    params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    initial_state = field.get_state()

    # Evolve for 10 steps
    for _ in range(10):
        state = field.evolve()
        assert state.time > initial_state.time
        assert 0.0 <= state.coherence <= 1.0
        assert state.entropy > 0.0
        assert state.energy > 0.0

    # Check that time advanced correctly
    final_state = field.get_state()
    expected_time = 10 * params.dt
    assert abs(final_state.time - expected_time) < 1e-6

    # Check history length
    history = field.get_history()
    assert len(history) == 11  # Initial + 10 evolution steps


def test_field_evolution_without_initialization():
    """Test that evolve raises error if field not initialized."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(RuntimeError, match="Field not initialized"):
        field.evolve()


def test_get_state_without_initialization():
    """Test that get_state raises error if field not initialized."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(RuntimeError, match="No state available"):
        field.get_state()


def test_field_history():
    """Test field history tracking."""

    params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    # Evolve for a few steps
    num_steps = 5
    for _ in range(num_steps):
        field.evolve()

    history = field.get_history()
    assert len(history) == 1 + num_steps  # Initial + num_steps

    # Check that history is ordered by time
    for i in range(1, len(history)):
        assert history[i].time > history[i - 1].time


def test_export_state():
    """Test exporting field state to dictionary."""

    params = QCMGParameters(
        grid_size=32,
        dt=0.01,
        coupling_strength=0.1,
        interaction_strength=0.05,
        thermal_noise=0.001,
        random_seed=42,
    )
    field = QuantacosmorphysigeneticField(params)
    field.initialize(mode="gaussian")

    # Evolve a bit
    for _ in range(3):
        field.evolve()

    export_data = field.export_state()

    # Check structure
    assert "parameters" in export_data
    assert "state" in export_data
    assert "history_length" in export_data

    # Check parameters
    assert export_data["parameters"]["grid_size"] == 32
    assert export_data["parameters"]["dt"] == 0.01
    assert export_data["parameters"]["coupling_strength"] == 0.1
    assert export_data["parameters"]["random_seed"] == 42

    # Check state
    assert "time" in export_data["state"]
    assert "coherence" in export_data["state"]
    assert "entropy" in export_data["state"]
    assert "energy" in export_data["state"]

    # Check history length
    assert export_data["history_length"] == 4  # Initial + 3 steps


def test_reproducibility_with_seed():
    """Test that using the same seed produces reproducible results."""

    params1 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field1 = QuantacosmorphysigeneticField(params1)
    field1.initialize(mode="gaussian")
    for _ in range(5):
        field1.evolve()
    state1 = field1.get_state()

    params2 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field2 = QuantacosmorphysigeneticField(params2)
    field2.initialize(mode="gaussian")
    for _ in range(5):
        field2.evolve()
    state2 = field2.get_state()

    # States should be identical with same seed
    assert abs(state1.coherence - state2.coherence) < 1e-10
    assert abs(state1.entropy - state2.entropy) < 1e-10
    assert abs(state1.energy - state2.energy) < 1e-10
    assert abs(state1.time - state2.time) < 1e-10


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""

    params1 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field1 = QuantacosmorphysigeneticField(params1)
    field1.initialize(mode="uniform")
    for _ in range(5):
        field1.evolve()
    state1 = field1.get_state()

    params2 = QCMGParameters(grid_size=32, dt=0.01, random_seed=123)
    field2 = QuantacosmorphysigeneticField(params2)
    field2.initialize(mode="uniform")
    for _ in range(5):
        field2.evolve()
    state2 = field2.get_state()

    # States should differ with different seeds
    # At least one metric should be significantly different
    differs = (
        abs(state1.coherence - state2.coherence) > 1e-6
        or abs(state1.entropy - state2.entropy) > 1e-6
        or abs(state1.energy - state2.energy) > 1e-6
    )
    assert differs

"""Tests for QuASIM simulation module."""

from __future__ import annotations

from quasim.simulation import (FieldState, QCMGParameters,
                               QuantacosmorphysigeneticField)


def test_qcmg_parameters_defaults():
    """Test QCMGParameters with default values."""

    params = QCMGParameters()
    assert params.coupling_strength == 1.0
    assert params.field_dimension == 3
    assert params.evolution_steps == 100
    assert params.precision == "fp64"
    assert params.temperature == 1.0


def test_qcmg_parameters_custom():
    """Test QCMGParameters with custom values."""

    params = QCMGParameters(
        coupling_strength=2.5,
        field_dimension=5,
        evolution_steps=50,
        precision="fp32",
        temperature=0.5,
    )
    assert params.coupling_strength == 2.5
    assert params.field_dimension == 5
    assert params.evolution_steps == 50
    assert params.precision == "fp32"
    assert params.temperature == 0.5


def test_field_state_creation():
    """Test FieldState creation and validation."""

    state = FieldState(
        field_values=[1 + 0j, 2 + 0j, 3 + 0j],
        momentum=[0.5 + 0j, 0.5 + 0j, 0.5 + 0j],
        energy=10.5,
        time=1.0,
    )
    assert len(state.field_values) == 3
    assert len(state.momentum) == 3
    assert state.energy == 10.5
    assert state.time == 1.0


def test_field_state_copy():
    """Test FieldState copy functionality."""

    state = FieldState(
        field_values=[1 + 0j, 2 + 0j],
        momentum=[0.5 + 0j, 0.5 + 0j],
        energy=5.0,
        time=2.0,
    )
    state_copy = state.copy()

    # Verify values are equal
    assert state_copy.field_values == state.field_values
    assert state_copy.momentum == state.momentum
    assert state_copy.energy == state.energy
    assert state_copy.time == state.time

    # Verify it's a deep copy
    assert state_copy is not state
    assert state_copy.field_values is not state.field_values


def test_qcmg_field_initialization():
    """Test QuantacosmorphysigeneticField initialization."""

    params = QCMGParameters(field_dimension=4)
    field = QuantacosmorphysigeneticField(params)

    assert field.parameters.field_dimension == 4
    assert not field._initialized
    assert len(field.history) == 0


def test_qcmg_field_initialize():
    """Test field initialization with default vacuum state."""

    params = QCMGParameters(field_dimension=3)
    field = QuantacosmorphysigeneticField(params)
    field.initialize()

    assert field._initialized
    assert len(field.state.field_values) == 3
    assert len(field.state.momentum) == 3
    assert field.state.energy == 0.0
    assert field.state.time == 0.0
    assert len(field.history) == 1


def test_qcmg_field_evolve():
    """Test field evolution."""

    params = QCMGParameters(field_dimension=2, coupling_strength=0.5)
    field = QuantacosmorphysigeneticField(params)

    # Initialize with non-zero momentum
    initial_state = FieldState(
        field_values=[1 + 0j, 0 + 1j],
        momentum=[0.1 + 0j, 0.1 + 0j],
        energy=0.0,
        time=0.0,
    )
    field.initialize(initial_state)

    # Evolve the field
    evolved_state = field.evolve(time_delta=0.1)

    assert evolved_state.time > 0.0
    assert evolved_state.energy >= 0.0
    assert len(field.history) == 2  # initial + evolved


def test_qcmg_field_simulate():
    """Test full simulation run."""

    params = QCMGParameters(field_dimension=2, evolution_steps=10)
    field = QuantacosmorphysigeneticField(params)
    field.initialize()

    trajectory = field.simulate(num_steps=5)

    assert len(trajectory) == 5
    assert all(isinstance(state, FieldState) for state in trajectory)
    assert trajectory[-1].time > trajectory[0].time


def test_qcmg_field_get_state():
    """Test getting current field state."""

    params = QCMGParameters(field_dimension=2)
    field = QuantacosmorphysigeneticField(params)
    field.initialize()

    state = field.get_state()
    assert isinstance(state, FieldState)
    assert state is not field.state  # Should be a copy


def test_qcmg_field_reset():
    """Test field reset functionality."""

    params = QCMGParameters(field_dimension=2)
    field = QuantacosmorphysigeneticField(params)
    field.initialize()
    field.evolve()

    # Reset the field
    field.reset()

    assert not field._initialized
    assert len(field.history) == 0
    assert len(field.state.field_values) == 0

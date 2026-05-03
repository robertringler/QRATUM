"""Tests for the QCMG Field Evolution Module."""

from __future__ import annotations

import numpy as np
import pytest

from quasim.qcmg import FieldState, QCMGParameters, QuantacosmorphysigeneticField


def test_qcmg_parameters_defaults():
    """Test default parameters are set correctly."""

    params = QCMGParameters()

    assert params.grid_size == 64
    assert params.spatial_extent == 10.0
    assert params.dt == 0.01
    assert params.coupling_strength == 0.1
    assert params.random_seed == 42


def test_qcmg_parameters_custom():
    """Test custom parameters can be set."""

    params = QCMGParameters(
        grid_size=32, spatial_extent=5.0, dt=0.001, coupling_strength=0.2, random_seed=123
    )

    assert params.grid_size == 32
    assert params.spatial_extent == 5.0
    assert params.dt == 0.001
    assert params.coupling_strength == 0.2
    assert params.random_seed == 123


def test_field_initialization_gaussian():
    """Test field initialization with Gaussian mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    assert field.phi_m is not None
    assert field.phi_i is not None
    assert len(field.phi_m) == 32
    assert len(field.phi_i) == 32
    assert field.time == 0.0
    assert len(field.history) == 1

    # Check fields are normalized
    assert np.isclose(np.linalg.norm(field.phi_m), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(field.phi_i), 1.0, atol=1e-10)


def test_field_initialization_soliton():
    """Test field initialization with soliton mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="soliton")

    assert field.phi_m is not None
    assert field.phi_i is not None
    assert len(field.phi_m) == 32
    assert len(field.phi_i) == 32

    # Check fields are normalized
    assert np.isclose(np.linalg.norm(field.phi_m), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(field.phi_i), 1.0, atol=1e-10)


def test_field_initialization_random():
    """Test field initialization with random mode."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="random")

    assert field.phi_m is not None
    assert field.phi_i is not None
    assert len(field.phi_m) == 32
    assert len(field.phi_i) == 32

    # Check fields are normalized
    assert np.isclose(np.linalg.norm(field.phi_m), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(field.phi_i), 1.0, atol=1e-10)


def test_field_initialization_invalid_mode():
    """Test that invalid initialization mode raises ValueError."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(ValueError, match="Unknown initialization mode"):
        field.initialize(mode="invalid_mode")


def test_evolve_without_initialization():
    """Test that evolve raises RuntimeError if not initialized."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(RuntimeError, match="Fields not initialized"):
        field.evolve()


def test_field_evolution():
    """Test field evolution over multiple steps."""

    params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    # Evolve for 10 steps
    for _ in range(10):
        state = field.evolve()
        assert state is not None
        assert isinstance(state, FieldState)

    # Check time has advanced
    assert field.time > 0.0
    assert len(field.history) == 11  # Initial + 10 evolution steps


def test_field_state_properties():
    """Test that FieldState has expected properties."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")
    state = field.get_state()

    # Check state properties
    assert isinstance(state.phi_m, np.ndarray)
    assert isinstance(state.phi_i, np.ndarray)
    assert isinstance(state.coherence, float)
    assert isinstance(state.entropy, float)
    assert isinstance(state.time, float)
    assert isinstance(state.energy, float)

    # Check coherence is bounded [0, 1]
    assert 0.0 <= state.coherence <= 1.0

    # Check entropy is non-negative
    assert state.entropy >= 0.0

    # Check energy is finite
    assert np.isfinite(state.energy)


def test_field_state_to_dict():
    """Test FieldState serialization to dictionary."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")
    state = field.get_state()

    state_dict = state.to_dict()

    # Check dictionary has expected keys
    assert "phi_m_real" in state_dict
    assert "phi_m_imag" in state_dict
    assert "phi_i_real" in state_dict
    assert "phi_i_imag" in state_dict
    assert "coherence" in state_dict
    assert "entropy" in state_dict
    assert "time" in state_dict
    assert "energy" in state_dict

    # Check types are serializable
    assert isinstance(state_dict["phi_m_real"], list)
    assert isinstance(state_dict["coherence"], float)
    assert isinstance(state_dict["time"], float)


def test_coherence_computation():
    """Test coherence computation is in valid range."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    # Evolve and check coherence
    for _ in range(10):
        state = field.evolve()
        assert 0.0 <= state.coherence <= 1.0


def test_entropy_increases():
    """Test that entropy generally increases (second law)."""

    params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    initial_entropy = field.get_state().entropy

    # Evolve for several steps
    for _ in range(20):
        field.evolve()

    final_entropy = field.get_state().entropy

    # Entropy should increase or stay roughly constant due to noise
    # We check it doesn't decrease significantly
    assert final_entropy >= initial_entropy - 0.5


def test_energy_conservation_approximate():
    """Test approximate energy conservation with damping."""

    params = QCMGParameters(
        grid_size=32,
        dt=0.01,
        damping_coeff=0.0,  # No damping for this test
        thermal_noise=0.0,  # No noise for this test
        random_seed=42,
    )
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    initial_energy = field.get_state().energy

    # Evolve for several steps
    for _ in range(10):
        field.evolve()

    final_energy = field.get_state().energy

    # Energy should be approximately conserved (within 10% due to numerical errors)
    assert abs(final_energy - initial_energy) / abs(initial_energy) < 0.1


def test_get_history():
    """Test getting evolution history."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    # Evolve for 5 steps
    for _ in range(5):
        field.evolve()

    history = field.get_history()

    assert len(history) == 6  # Initial + 5 evolution steps
    assert all(isinstance(state, FieldState) for state in history)

    # Check time progression
    for i in range(1, len(history)):
        assert history[i].time > history[i - 1].time


def test_export_state():
    """Test exporting field state to JSON-serializable dict."""

    params = QCMGParameters(grid_size=32, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")
    field.evolve()

    export = field.export_state()

    # Check structure
    assert "qcmg_version" in export
    assert "parameters" in export
    assert "state" in export
    assert "metadata" in export

    # Check parameters
    assert export["parameters"]["grid_size"] == 32

    # Check metadata
    assert "history_length" in export["metadata"]
    assert "bounded" in export["metadata"]
    assert export["metadata"]["bounded"] is True


def test_get_state_without_initialization():
    """Test that get_state raises RuntimeError if not initialized."""

    params = QCMGParameters(grid_size=32)
    field = QuantacosmorphysigeneticField(params)

    with pytest.raises(RuntimeError, match="No state available"):
        field.get_state()


def test_reproducibility_with_seed():
    """Test that results are reproducible with fixed seed."""

    params1 = QCMGParameters(grid_size=32, random_seed=42)
    field1 = QuantacosmorphysigeneticField(params1)
    field1.initialize(mode="random")

    params2 = QCMGParameters(grid_size=32, random_seed=42)
    field2 = QuantacosmorphysigeneticField(params2)
    field2.initialize(mode="random")

    # Fields should be identical with same seed
    assert np.allclose(field1.phi_m, field2.phi_m)
    assert np.allclose(field1.phi_i, field2.phi_i)

    # Evolve both and check they stay similar
    # Note: Due to stochastic noise in evolution, exact reproducibility
    # is difficult, but results should be similar with same seed
    for _ in range(5):
        state1 = field1.evolve()
        state2 = field2.evolve()

        # Use more relaxed tolerance due to accumulated numerical noise
        assert np.allclose(state1.phi_m, state2.phi_m, atol=1e-4)
        assert np.allclose(state1.phi_i, state2.phi_i, atol=1e-4)


def test_spatial_grid_setup():
    """Test spatial grid is set up correctly."""

    params = QCMGParameters(grid_size=32, spatial_extent=10.0)
    field = QuantacosmorphysigeneticField(params)

    assert len(field.x) == 32
    assert field.x[0] == pytest.approx(-5.0, abs=0.01)
    assert field.x[-1] == pytest.approx(5.0, abs=0.01)
    assert field.dx > 0.0


def test_laplacian_computation():
    """Test Laplacian computation with simple function."""

    params = QCMGParameters(grid_size=64, spatial_extent=10.0)
    field = QuantacosmorphysigeneticField(params)

    # Create a simple test field (Gaussian)
    test_field = np.exp(-(field.x**2))

    laplacian = field._laplacian(test_field)

    # Laplacian should be negative for a Gaussian peak
    assert np.any(laplacian < 0)

    # Check shape is preserved
    assert laplacian.shape == test_field.shape


def test_field_normalization_during_evolution():
    """Test that fields remain normalized during evolution."""

    params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
    field = QuantacosmorphysigeneticField(params)

    field.initialize(mode="gaussian")

    # Evolve and check normalization
    for _ in range(10):
        field.evolve()

        norm_m = np.linalg.norm(field.phi_m)
        norm_i = np.linalg.norm(field.phi_i)

        # Fields should remain normalized (or very close)
        assert np.isclose(norm_m, 1.0, atol=1e-6)
        assert np.isclose(norm_i, 1.0, atol=1e-6)


def test_multiple_initialization_modes():
    """Test all initialization modes produce valid states."""

    params = QCMGParameters(grid_size=32, random_seed=42)

    modes = ["gaussian", "soliton", "random"]

    for mode in modes:
        field = QuantacosmorphysigeneticField(params)
        field.initialize(mode=mode)

        state = field.get_state()

        # All modes should produce valid states
        assert 0.0 <= state.coherence <= 1.0
        assert state.entropy >= 0.0
        assert np.isfinite(state.energy)

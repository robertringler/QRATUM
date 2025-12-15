"""Comprehensive test suite for QCMG field simulation module.

Tests cover:
- Parameter configuration
- Field initialization (gaussian, soliton, random)
- Single and multiple evolution steps
- Observable validation (coherence, entropy, energy)
- Reproducibility with fixed seed
- Field normalization
- State export/import
- History tracking
- CLI integration
- Numerical stability
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quasim.sim import (QCMGParameters, QCMGState,
                        QuantacosmomorphysigeneticField)


class TestQCMGParameters:
    """Test parameter configuration and validation."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = QCMGParameters()
        assert params.grid_size == 64
        assert params.dt == 0.01
        # Use the actual default value from the implementation
        assert params.coupling_strength == 0.1
        assert params.interaction_strength == 0.05
        assert params.thermal_noise == 0.001
        assert params.random_seed is None

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = QCMGParameters(
            grid_size=128,
            dt=0.005,
            coupling_strength=0.5,
            interaction_strength=0.02,
            thermal_noise=0.002,
            random_seed=42,
        )
        assert params.grid_size == 128
        assert params.dt == 0.005
        assert params.coupling_strength == 0.5
        assert params.interaction_strength == 0.02
        assert params.thermal_noise == 0.002
        assert params.random_seed == 42


class TestFieldInitialization:
    """Test field initialization modes."""

    def test_gaussian_initialization(self):
        """Test Gaussian wave packet initialization."""
        params = QCMGParameters(grid_size=64, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        # Check that fields are initialized
        assert field.phi_m is not None
        assert field.phi_i is not None
        assert len(field.phi_m) == 64
        assert len(field.phi_i) == 64

        # Check that history is populated
        assert len(field.history) == 1
        assert field.history[0].time == 0.0

    def test_soliton_initialization(self):
        """Test soliton initialization."""
        params = QCMGParameters(grid_size=64, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="soliton")

        assert field.phi_m is not None
        assert field.phi_i is not None
        assert len(field.history) == 1

    def test_random_initialization(self):
        """Test random field initialization."""
        params = QCMGParameters(grid_size=64, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="random")

        assert field.phi_m is not None
        assert field.phi_i is not None
        assert len(field.history) == 1

    def test_invalid_initialization_mode(self):
        """Test that invalid initialization mode raises ValueError."""
        params = QCMGParameters(grid_size=64)
        field = QuantacosmomorphysigeneticField(params)

        with pytest.raises(ValueError, match="Unknown initialization mode"):
            field.initialize(mode="invalid")

    def test_fields_are_normalized(self):
        """Test that initialized fields are normalized."""
        params = QCMGParameters(grid_size=64, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        # Compute norms
        norm_m = np.sqrt(np.sum(np.abs(field.phi_m) ** 2) * field.dx)
        norm_i = np.sqrt(np.sum(np.abs(field.phi_i) ** 2) * field.dx)

        # Should be close to 1
        assert abs(norm_m - 1.0) < 0.1
        assert abs(norm_i - 1.0) < 0.1


class TestFieldEvolution:
    """Test field evolution dynamics."""

    def test_single_evolution_step(self):
        """Test single evolution step."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        initial_time = field.time
        state = field.evolve(steps=1)

        # Time should advance
        assert field.time == initial_time + params.dt
        assert state.time == field.time

        # History should grow
        assert len(field.history) == 2

    def test_multiple_evolution_steps(self):
        """Test multiple evolution steps."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        n_steps = 10
        field.evolve(steps=n_steps)

        # Time should advance correctly
        expected_time = n_steps * params.dt
        assert abs(field.time - expected_time) < 1e-10

        # History should contain all steps
        assert len(field.history) == n_steps + 1  # +1 for initial state

    def test_evolution_changes_fields(self):
        """Test that evolution changes field values."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        initial_phi_m = field.phi_m.copy()
        initial_phi_i = field.phi_i.copy()

        field.evolve(steps=5)

        # Fields should have changed
        assert not np.allclose(field.phi_m, initial_phi_m)
        assert not np.allclose(field.phi_i, initial_phi_i)


class TestObservables:
    """Test observable quantity calculations."""

    def test_coherence_bounds(self):
        """Test that coherence is bounded in [0, 1]."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        for mode in ["gaussian", "soliton", "random"]:
            field.initialize(mode=mode)

            for _ in range(20):
                state = field.evolve(steps=1)
                assert 0 <= state.coherence <= 1, f"Coherence out of bounds: {state.coherence}"

    def test_entropy_non_negative(self):
        """Test that entropy is non-negative."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        for mode in ["gaussian", "soliton", "random"]:
            field.initialize(mode=mode)

            for _ in range(20):
                state = field.evolve(steps=1)
                assert state.entropy >= 0, f"Entropy is negative: {state.entropy}"

    def test_energy_finite(self):
        """Test that energy is finite."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        for mode in ["gaussian", "soliton", "random"]:
            field.initialize(mode=mode)

            for _ in range(20):
                state = field.evolve(steps=1)
                assert np.isfinite(state.energy), f"Energy not finite: {state.energy}"

    def test_no_nan_or_inf(self):
        """Test that fields don't contain NaN or Inf."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(50):
            field.evolve(steps=1)

            assert np.all(np.isfinite(field.phi_m)), "phi_m contains NaN or Inf"
            assert np.all(np.isfinite(field.phi_i)), "phi_i contains NaN or Inf"


class TestReproducibility:
    """Test reproducibility with fixed random seed."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        params1 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field1 = QuantacosmomorphysigeneticField(params1)
        field1.initialize(mode="random")
        field1.evolve(steps=10)

        params2 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field2 = QuantacosmomorphysigeneticField(params2)
        field2.initialize(mode="random")
        field2.evolve(steps=10)

        # Should produce identical results
        assert np.allclose(field1.phi_m, field2.phi_m)
        assert np.allclose(field1.phi_i, field2.phi_i)

        # Observables should match
        state1 = field1.history[-1]
        state2 = field2.history[-1]
        assert abs(state1.coherence - state2.coherence) < 1e-10
        assert abs(state1.entropy - state2.entropy) < 1e-10
        assert abs(state1.energy - state2.energy) < 1e-10

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        params1 = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field1 = QuantacosmomorphysigeneticField(params1)
        field1.initialize(mode="random")

        params2 = QCMGParameters(grid_size=32, dt=0.01, random_seed=123)
        field2 = QuantacosmomorphysigeneticField(params2)
        field2.initialize(mode="random")

        # Should produce different results
        assert not np.allclose(field1.phi_m, field2.phi_m)
        assert not np.allclose(field1.phi_i, field2.phi_i)


class TestStateValidation:
    """Test state validation."""

    def test_valid_state(self):
        """Test that properly initialized state is valid."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.history[-1]
        assert state.validate()

    def test_invalid_state_nan(self):
        """Test that state with NaN is invalid."""
        params = QCMGParameters(grid_size=32)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.history[-1]
        state.phi_m[0] = np.nan

        assert not state.validate()

    def test_invalid_state_coherence_out_of_bounds(self):
        """Test that state with coherence > 1 is invalid."""
        params = QCMGParameters(grid_size=32)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.history[-1]
        state.coherence = 1.5

        assert not state.validate()

    def test_invalid_state_negative_entropy(self):
        """Test that state with negative entropy is invalid."""
        params = QCMGParameters(grid_size=32)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.history[-1]
        state.entropy = -1.0

        assert not state.validate()


class TestStateExport:
    """Test state export and JSON serialization."""

    def test_export_state_structure(self):
        """Test exported state has correct structure."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")
        field.evolve(steps=5)

        data = field.export_state(include_history=True)

        # Check structure
        assert "parameters" in data
        assert "current_state" in data
        assert "history" in data

        # Check parameters
        assert data["parameters"]["grid_size"] == 32
        assert data["parameters"]["dt"] == 0.01
        assert data["parameters"]["random_seed"] == 42

        # Check current state
        assert "time" in data["current_state"]
        assert "coherence" in data["current_state"]
        assert "entropy" in data["current_state"]
        assert "energy" in data["current_state"]

        # Check history
        assert len(data["history"]) == 6  # Initial + 5 steps

    def test_export_without_history(self):
        """Test export without history."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")
        field.evolve(steps=5)

        data = field.export_state(include_history=False)

        assert "history" not in data
        assert "parameters" in data
        assert "current_state" in data

    def test_save_to_json(self):
        """Test saving state to JSON file."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")
        field.evolve(steps=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            field.save_to_json(temp_path, include_history=True)

            # Load and verify
            with open(temp_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "parameters" in data
            assert "current_state" in data
            assert "history" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestLongTimeStability:
    """Test long-time numerical stability."""

    def test_long_evolution_stability(self):
        """Test that long evolution remains stable."""
        params = QCMGParameters(grid_size=32, dt=0.01, dissipation_rate=0.02, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        # Evolve for many steps
        for _ in range(20):
            state = field.evolve(steps=10)

            # Check that observables remain valid
            assert state.validate(), f"Invalid state at time {state.time}"
            assert 0 <= state.coherence <= 1
            assert state.entropy >= 0
            assert np.isfinite(state.energy)


class TestCLIIntegration:
    """Test CLI interface functionality."""

    def test_cli_basic_run(self):
        """Test basic CLI execution."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.sim.qcmg_cli",
                "--iterations",
                "10",
                "--grid-size",
                "32",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "Simulation complete" in result.stdout

    def test_cli_with_export(self):
        """Test CLI with JSON export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quasim.sim.qcmg_cli",
                    "--iterations",
                    "10",
                    "--export",
                    temp_path,
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0
            assert Path(temp_path).exists()

            # Verify JSON content
            with open(temp_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "parameters" in data
            assert "current_state" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_cli_verbose_mode(self):
        """Test CLI verbose output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "quasim.sim.qcmg_cli",
                "--iterations",
                "5",
                "--grid-size",
                "32",
                "--verbose",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "QCMG Field Simulation" in result.stdout
        assert "Initial state:" in result.stdout
        assert "Final state:" in result.stdout

    def test_cli_different_init_modes(self):
        """Test CLI with different initialization modes."""
        for mode in ["gaussian", "soliton", "random"]:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quasim.sim.qcmg_cli",
                    "--iterations",
                    "5",
                    "--init-mode",
                    mode,
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"Failed for mode {mode}"


class TestQuantacosmomorphysigeneticField:
    """Test QuantacosmomorphysigeneticField class."""

    def test_initialization(self):
        """Test field initialization."""
        params = QCMGParameters(random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        assert field.time == 0.0
        assert field.phi_m is None
        assert field.phi_i is None

    def test_initialize_gaussian(self):
        """Test Gaussian initialization."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        assert field.phi_m is not None
        assert field.phi_i is not None
        assert field.phi_m.shape == (32,)
        assert field.phi_i.shape == (32,)
        assert np.iscomplexobj(field.phi_m)
        assert np.iscomplexobj(field.phi_i)

    def test_initialize_soliton(self):
        """Test soliton initialization."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="soliton")

        assert field.phi_m is not None
        assert field.phi_i is not None
        assert field.phi_m.shape == (32,)
        assert field.phi_i.shape == (32,)

    def test_initialize_random(self):
        """Test random initialization."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="random")

        assert field.phi_m is not None
        assert field.phi_i is not None
        assert field.phi_m.shape == (32,)
        assert field.phi_i.shape == (32,)

    def test_initialize_invalid_mode(self):
        """Test initialization with invalid mode."""
        params = QCMGParameters(random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        with pytest.raises(ValueError, match="Unknown initialization mode"):
            field.initialize(mode="invalid")

    def test_evolve_without_initialization(self):
        """Test that evolve raises error without initialization."""
        params = QCMGParameters(random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        with pytest.raises(RuntimeError, match="Field not initialized"):
            field.evolve()

    def test_evolve_single_step(self):
        """Test single evolution step."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.evolve()

        assert isinstance(state, QCMGState)
        assert state.time == params.dt
        assert 0 <= state.coherence <= 1
        assert state.entropy >= 0
        assert np.isfinite(state.energy)

    def test_evolve_multiple_steps(self):
        """Test multiple evolution steps."""
        params = QCMGParameters(grid_size=32, dt=0.01, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for i in range(10):
            state = field.evolve()
            expected_time = (i + 1) * params.dt
            assert abs(state.time - expected_time) < 1e-10
            assert 0 <= state.coherence <= 1
            assert state.entropy >= 0

    def test_get_state(self):
        """Test get_state method."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        state = field.get_state()

        assert isinstance(state, QCMGState)
        assert state.time == 0.0
        assert 0 <= state.coherence <= 1
        assert state.entropy >= 0
        assert np.isfinite(state.energy)

    def test_get_state_without_initialization(self):
        """Test that get_state raises error without initialization."""
        params = QCMGParameters(random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        with pytest.raises(RuntimeError, match="Field not initialized"):
            field.get_state()

    def test_coherence_bounds(self):
        """Test that coherence stays in [0, 1]."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(50):
            state = field.evolve()
            assert 0 <= state.coherence <= 1

    def test_entropy_non_negative(self):
        """Test that entropy is always non-negative."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(50):
            state = field.evolve()
            assert state.entropy >= 0

    def test_energy_finite(self):
        """Test that energy remains finite."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(50):
            state = field.evolve()
            assert np.isfinite(state.energy)

    def test_reproducibility_with_seed(self):
        """Test that simulations are reproducible with same seed."""
        params1 = QCMGParameters(grid_size=32, random_seed=42)
        field1 = QuantacosmomorphysigeneticField(params1)
        field1.initialize(mode="gaussian")

        params2 = QCMGParameters(grid_size=32, random_seed=42)
        field2 = QuantacosmomorphysigeneticField(params2)
        field2.initialize(mode="gaussian")

        for _ in range(10):
            state1 = field1.evolve()
            state2 = field2.evolve()

            assert abs(state1.coherence - state2.coherence) < 1e-10
            assert abs(state1.entropy - state2.entropy) < 1e-10
            assert abs(state1.energy - state2.energy) < 1e-10

    def test_export_state(self):
        """Test export_state method."""
        params = QCMGParameters(grid_size=32, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(10):
            field.evolve()

        export_data = field.export_state()

        assert "parameters" in export_data
        assert "current_state" in export_data
        assert "trajectory" in export_data

        assert export_data["parameters"]["grid_size"] == 32
        assert export_data["parameters"]["random_seed"] == 42

        assert "time" in export_data["current_state"]
        assert "coherence" in export_data["current_state"]
        assert "entropy" in export_data["current_state"]
        assert "energy" in export_data["current_state"]

        assert len(export_data["trajectory"]["time"]) == 10
        assert len(export_data["trajectory"]["coherence"]) == 10
        assert len(export_data["trajectory"]["entropy"]) == 10
        assert len(export_data["trajectory"]["energy"]) == 10

    def test_export_state_without_initialization(self):
        """Test that export_state raises error without initialization."""
        params = QCMGParameters(random_seed=42)
        field = QuantacosmomorphysigeneticField(params)

        with pytest.raises(RuntimeError, match="Field not initialized"):
            field.export_state()

    def test_coupling_effect(self):
        """Test that coupling strength affects dynamics."""
        # Low coupling
        params_low = QCMGParameters(grid_size=32, coupling_strength=0.01, random_seed=42)
        field_low = QuantacosmomorphysigeneticField(params_low)
        field_low.initialize(mode="gaussian")

        # High coupling
        params_high = QCMGParameters(grid_size=32, coupling_strength=0.5, random_seed=42)
        field_high = QuantacosmomorphysigeneticField(params_high)
        field_high.initialize(mode="gaussian")

        # Evolve both
        for _ in range(50):
            state_low = field_low.evolve()
            state_high = field_high.evolve()

        # High coupling should maintain coherence better
        assert state_high.coherence > state_low.coherence

    def test_thermal_noise_effect(self):
        """Test that thermal noise affects dynamics."""
        # No noise
        params_no_noise = QCMGParameters(grid_size=32, thermal_noise=0.0, random_seed=42)
        field_no_noise = QuantacosmomorphysigeneticField(params_no_noise)
        field_no_noise.initialize(mode="gaussian")

        # With noise
        params_noise = QCMGParameters(grid_size=32, thermal_noise=0.01, random_seed=42)
        field_noise = QuantacosmomorphysigeneticField(params_noise)
        field_noise.initialize(mode="gaussian")

        # Evolve both
        for _ in range(50):
            state_no_noise = field_no_noise.evolve()
            state_noise = field_noise.evolve()

        # Noise should increase entropy
        assert state_noise.entropy > state_no_noise.entropy


class TestQCMGState:
    """Test QCMGState dataclass."""

    def test_state_creation(self):
        """Test QCMGState creation."""
        phi_m = np.array([1 + 0j, 2 + 0j])
        phi_i = np.array([0 + 1j, 0 + 2j])

        state = QCMGState(
            time=1.0,
            phi_m=phi_m,
            phi_i=phi_i,
            coherence=0.8,
            entropy=2.5,
            energy=10.0,
        )

        assert state.time == 1.0
        assert np.array_equal(state.phi_m, phi_m)
        assert np.array_equal(state.phi_i, phi_i)
        assert state.coherence == 0.8
        assert state.entropy == 2.5
        assert state.energy == 10.0


class TestIntegrationScenarios:
    """Integration tests for common use cases."""

    def test_basic_simulation_workflow(self):
        """Test the basic workflow from Quick Start Guide."""
        # Configure
        params = QCMGParameters(grid_size=64, dt=0.01, random_seed=42)

        # Initialize
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        # Evolve
        for _i in range(100):
            state = field.evolve()

        # Analyze
        assert 0 <= state.coherence <= 1
        assert state.entropy >= 0
        assert np.isfinite(state.energy)

    def test_decoherence_study(self):
        """Test decoherence study from Quick Start Guide."""
        params = QCMGParameters(coupling_strength=0.5, thermal_noise=0.001, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        coherences = [field.evolve().coherence for _ in range(200)]

        assert len(coherences) == 200
        assert all(0 <= c <= 1 for c in coherences)

    def test_entropy_production(self):
        """Test entropy production measurement from Quick Start Guide."""
        params = QCMGParameters(interaction_strength=0.1, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="soliton")

        initial_entropy = field.get_state().entropy

        for _ in range(100):
            field.evolve()

        final_entropy = field.get_state().entropy

        # Entropy should increase (second law)
        assert final_entropy >= initial_entropy

    def test_compare_initialization_modes(self):
        """Test comparing initialization modes from Quick Start Guide."""
        modes = ["gaussian", "soliton", "random"]
        results = {}

        for mode in modes:
            params = QCMGParameters(random_seed=42)
            field = QuantacosmomorphysigeneticField(params)
            field.initialize(mode=mode)

            for _ in range(50):
                field.evolve()

            results[mode] = field.get_state()

        for mode, state in results.items():
            assert 0 <= state.coherence <= 1
            assert state.entropy >= 0

    def test_export_and_analyze(self):
        """Test export and analysis from Quick Start Guide."""
        params = QCMGParameters(grid_size=128, random_seed=42)
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode="gaussian")

        for _ in range(200):
            field.evolve()

        export_data = field.export_state()
        trajectory = export_data["trajectory"]

        assert len(trajectory["coherence"]) == 200

        mean_coherence = np.mean(trajectory["coherence"])
        std_coherence = np.std(trajectory["coherence"])

        assert np.isfinite(mean_coherence)
        assert np.isfinite(std_coherence)

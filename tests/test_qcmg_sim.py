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
from pathlib import Path
import tempfile

import numpy as np
import pytest

from quasim.sim import (
    QCMGParameters,
    FieldState,
    QuantacosmomorphysigeneticField,
)


class TestQCMGParameters:
    """Test parameter configuration and validation."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = QCMGParameters()
        assert params.grid_size == 64
        assert params.dt == 0.01
        assert params.coupling_strength == 1.0
        assert params.dissipation_rate == 0.01
        assert params.random_seed is None
    
    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        params = QCMGParameters(
            grid_size=128,
            dt=0.005,
            coupling_strength=0.5,
            dissipation_rate=0.02,
            random_seed=42,
        )
        assert params.grid_size == 128
        assert params.dt == 0.005
        assert params.coupling_strength == 0.5
        assert params.dissipation_rate == 0.02
        assert params.random_seed == 42
    
    def test_invalid_grid_size(self):
        """Test that invalid grid_size raises ValueError."""
        with pytest.raises(ValueError, match="grid_size must be positive"):
            QCMGParameters(grid_size=0)
        
        with pytest.raises(ValueError, match="grid_size must be positive"):
            QCMGParameters(grid_size=-10)
    
    def test_invalid_dt(self):
        """Test that invalid dt raises ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            QCMGParameters(dt=0)
        
        with pytest.raises(ValueError, match="dt must be positive"):
            QCMGParameters(dt=-0.01)
    
    def test_invalid_coupling(self):
        """Test that invalid coupling_strength raises ValueError."""
        with pytest.raises(ValueError, match="coupling_strength must be non-negative"):
            QCMGParameters(coupling_strength=-1.0)
    
    def test_invalid_dissipation(self):
        """Test that invalid dissipation_rate raises ValueError."""
        with pytest.raises(ValueError, match="dissipation_rate must be non-negative"):
            QCMGParameters(dissipation_rate=-0.01)


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
        state = field.evolve(steps=n_steps)
        
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
            with open(temp_path, "r", encoding="utf-8") as f:
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
        params = QCMGParameters(
            grid_size=32,
            dt=0.01,
            dissipation_rate=0.02,
            random_seed=42
        )
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
            with open(temp_path, "r", encoding="utf-8") as f:
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

"""Tests for Lattice Field Model."""

import numpy as np
import pytest

from compiled_realities.models.lattice_field import LatticeFieldModel


def test_lattice_model_initialization():
    """Test Lattice Field model initialization."""
    config = {
        'n_sites': 32,
        'cutoff_scale': 10,
        'dispersion_strength': 0.5,
        'flow_velocity_ratio': 0.9,
        'dissipation_rate': 0.01,
        'n_timesteps': 50,
        'dt': 0.01,
    }
    
    model = LatticeFieldModel(config, seed=42)
    
    assert model.n_sites == 32
    assert model.cutoff == 10
    assert model.dispersion == 0.5
    assert model.flow_vel == 0.9
    assert model.dissipation == 0.01


def test_lattice_simulation_runs():
    """Test that Lattice simulation runs without errors."""
    config = {
        'n_sites': 16,
        'cutoff_scale': 10,
        'dispersion_strength': 0.0,
        'flow_velocity_ratio': 0.5,
        'dissipation_rate': 0.01,
        'n_timesteps': 20,
        'dt': 0.01,
    }
    
    model = LatticeFieldModel(config, seed=42)
    results = model.run_simulation(intervention=0)
    
    assert 'observables_t1' in results
    assert 'effective_metric_data' in results
    assert isinstance(results['observables_t1'], (float, np.floating))


def test_lattice_twin_simulations():
    """Test twin simulation with I=0 and I=1."""
    config = {
        'n_sites': 16,
        'cutoff_scale': 10,
        'dispersion_strength': 0.5,
        'flow_velocity_ratio': 0.8,
        'dissipation_rate': 0.001,
        'n_timesteps': 20,
        'dt': 0.01,
    }
    
    model = LatticeFieldModel(config, seed=42)
    results_I0, results_I1 = model.run_twin_simulations()
    
    assert 'observables_t1' in results_I0
    assert 'observables_t1' in results_I1
    assert isinstance(results_I0['observables_t1'], (float, np.floating))
    assert isinstance(results_I1['observables_t1'], (float, np.floating))


def test_lattice_effective_metric():
    """Test effective metric computation."""
    config = {
        'n_sites': 32,
        'cutoff_scale': 100,
        'dispersion_strength': 1.0,
        'flow_velocity_ratio': 0.95,
        'dissipation_rate': 0.01,
        'n_timesteps': 20,
        'dt': 0.01,
    }
    
    model = LatticeFieldModel(config, seed=42)
    metric_data = model.effective_metric_data
    
    assert 'c_eff' in metric_data
    assert 'dispersion_deviation' in metric_data
    assert 'attenuation_margin' in metric_data
    assert 'reconstruction_error' in metric_data
    assert isinstance(metric_data['c_eff'], (float, np.floating))


def test_lattice_observable_extraction():
    """Test observable extraction at t1."""
    config = {
        'n_sites': 16,
        'cutoff_scale': 10,
        'dispersion_strength': 0.0,
        'flow_velocity_ratio': 0.7,
        'dissipation_rate': 0.001,
        'n_timesteps': 20,
        'dt': 0.01,
    }
    
    model = LatticeFieldModel(config, seed=42)
    results = model.run_simulation(intervention=0)
    obs = model.get_observable_t1(results)
    
    assert isinstance(obs, (float, np.floating))
    assert obs >= 0  # Energy should be non-negative

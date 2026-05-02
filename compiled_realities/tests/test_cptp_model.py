"""Tests for CPTP Process-Tensor Model."""

import numpy as np

from compiled_realities.models.cptp_process_tensor import \
    CPTPProcessTensorModel


def test_cptp_model_initialization():
    """Test CPTP model initialization."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 50,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)

    assert model.dim == 2
    assert model.gamma == 0.1
    assert model.t1 == 5
    assert model.t2 == 10
    assert model.seed == 42


def test_cptp_simulation_runs():
    """Test that CPTP simulation runs without errors."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 30,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results = model.run_simulation(intervention=0)

    assert 'observables_t1' in results
    assert 'full_trajectory' in results
    assert 'metadata' in results
    assert isinstance(results['observables_t1'], (float, np.floating))


def test_cptp_twin_simulations():
    """Test twin simulation with I=0 and I=1."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 30,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results_I0, results_I1 = model.run_twin_simulations()

    assert 'observables_t1' in results_I0
    assert 'observables_t1' in results_I1

    # Observables may differ after intervention
    # but should have valid values
    assert isinstance(results_I0['observables_t1'], (float, np.floating))
    assert isinstance(results_I1['observables_t1'], (float, np.floating))


def test_cptp_observable_extraction():
    """Test observable extraction at t1."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 30,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results = model.run_simulation(intervention=0)
    obs = model.get_observable_t1(results)

    assert isinstance(obs, (float, np.floating))
    assert 0 <= obs <= 1  # Should be a probability


def test_cptp_dimension_4():
    """Test CPTP model with dimension 4."""
    config = {
        'dimension': 4,
        'environment_coupling': 0.05,
        't1_t2_separation': 3,
        'n_timesteps': 20,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=123)
    results = model.run_simulation(intervention=1)

    assert results['metadata']['dimension'] == 4
    assert isinstance(results['observables_t1'], (float, np.floating))

"""Tests for diagnostics."""

import numpy as np

from compiled_realities.diagnostics.chi_effective import (
    compute_chi_effective, compute_chi_statistics)
from compiled_realities.diagnostics.s_lab import (check_s_lab_significance,
                                                  compute_s_lab)
from compiled_realities.diagnostics.validity import check_validity
from compiled_realities.models.cptp_process_tensor import \
    CPTPProcessTensorModel
from compiled_realities.models.lattice_field import LatticeFieldModel


def test_s_lab_cptp():
    """Test S_lab computation for CPTP model."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 30,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    s_lab_result = compute_s_lab(model, n_samples=10)

    assert 's_lab' in s_lab_result
    assert 's_lab_stderr' in s_lab_result
    assert 'p_value' in s_lab_result
    assert isinstance(s_lab_result['s_lab'], float)
    assert s_lab_result['s_lab'] >= 0


def test_s_lab_lattice():
    """Test S_lab computation for Lattice model."""
    config = {
        'n_sites': 16,
        'cutoff_scale': 10,
        'dispersion_strength': 0.5,
        'flow_velocity_ratio': 0.8,
        'dissipation_rate': 0.01,
        'n_timesteps': 20,
        'dt': 0.01,
    }

    model = LatticeFieldModel(config, seed=42)
    s_lab_result = compute_s_lab(model, n_samples=10)

    assert 's_lab' in s_lab_result
    assert isinstance(s_lab_result['s_lab'], float)


def test_s_lab_significance():
    """Test S_lab significance checking."""
    s_lab_result = {
        's_lab': 0.001,
        's_lab_stderr': 0.0001,
    }

    # Should be significant at 5 sigma (10 sigma here)
    is_sig = check_s_lab_significance(s_lab_result, threshold_sigma=5.0)
    assert is_sig

    # Should not be significant at 20 sigma
    is_sig = check_s_lab_significance(s_lab_result, threshold_sigma=20.0)
    assert not is_sig


def test_chi_effective_lattice():
    """Test chi computation for Lattice model."""
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
    results = model.run_simulation(intervention=0)
    chi_result = compute_chi_effective(model, results)

    assert 'chi' in chi_result
    assert 'flow_velocity' in chi_result
    assert isinstance(chi_result['chi'], (float, np.floating))


def test_chi_effective_cptp_not_applicable():
    """Test that chi returns None for CPTP model."""
    config = {
        'dimension': 2,
        'environment_coupling': 0.1,
        't1_t2_separation': 5,
        'n_timesteps': 30,
        'dt': 0.1,
    }

    model = CPTPProcessTensorModel(config, seed=42)
    results = model.run_simulation(intervention=0)
    chi_result = compute_chi_effective(model, results)

    assert chi_result['chi'] is None


def test_chi_statistics():
    """Test chi statistics computation."""
    chi_results = [
        {'chi': 0.5},
        {'chi': -0.2},
        {'chi': 0.1},
        {'chi': None},
        {'chi': -0.5},
    ]

    stats = compute_chi_statistics(chi_results)

    assert 'chi_mean' in stats
    assert 'n_negative' in stats
    assert stats['n_valid'] == 4
    assert stats['n_negative'] == 2


def test_validity_check():
    """Test validity certificate checking."""
    metric_data = {
        'dispersion_deviation': 0.05,
        'attenuation_margin': 2.0,
        'reconstruction_error': 0.08,
    }

    thresholds = {
        'dispersion_deviation_max': 0.10,
        'attenuation_margin_min': 1.5,
        'reconstruction_error_max': 0.10,
    }

    validity = check_validity(metric_data, thresholds)

    assert validity['all_pass'] is True
    assert validity['dispersion_deviation_pass'] is True
    assert validity['attenuation_margin_pass'] is True
    assert validity['reconstruction_error_pass'] is True


def test_validity_check_fails():
    """Test validity check failure."""
    metric_data = {
        'dispersion_deviation': 0.15,  # Exceeds threshold
        'attenuation_margin': 2.0,
        'reconstruction_error': 0.08,
    }

    thresholds = {
        'dispersion_deviation_max': 0.10,
        'attenuation_margin_min': 1.5,
        'reconstruction_error_max': 0.10,
    }

    validity = check_validity(metric_data, thresholds)

    assert validity['all_pass'] is False
    assert validity['dispersion_deviation_pass'] is False

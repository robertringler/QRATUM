"""Validity certificate for chi reporting."""

from typing import Dict, Any


def check_validity(metric_data: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Check validity certificate for chi reporting.
    
    Chi can only be reported if validity checks pass:
    - Dispersion deviation < threshold
    - Attenuation margin > threshold
    - Reconstruction error < threshold
    
    Args:
        metric_data: Effective metric data from lattice model
        thresholds: Dictionary with validity thresholds
        
    Returns:
        Dictionary with validity status and individual checks
    """
    dispersion_dev = metric_data.get('dispersion_deviation', float('inf'))
    attenuation = metric_data.get('attenuation_margin', 0.0)
    recon_error = metric_data.get('reconstruction_error', float('inf'))
    
    dispersion_dev_max = thresholds.get('dispersion_deviation_max', 0.10)
    attenuation_min = thresholds.get('attenuation_margin_min', 1.5)
    recon_error_max = thresholds.get('reconstruction_error_max', 0.10)
    
    checks = {
        'dispersion_deviation_pass': dispersion_dev < dispersion_dev_max,
        'attenuation_margin_pass': attenuation > attenuation_min,
        'reconstruction_error_pass': recon_error < recon_error_max,
    }
    
    checks['all_pass'] = all(checks.values())
    
    checks['values'] = {
        'dispersion_deviation': float(dispersion_dev),
        'attenuation_margin': float(attenuation),
        'reconstruction_error': float(recon_error),
    }
    
    checks['thresholds'] = thresholds
    
    return checks

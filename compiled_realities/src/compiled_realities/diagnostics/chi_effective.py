"""Effective-time loop diagnostic (χ).

χ(θ) = inf_{γ∈C} ∮_γ g_eff(θ)_{μν} dx^μ dx^ν
"""

from typing import Any, Dict

import numpy as np


def compute_chi_effective(model, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute effective-time loop diagnostic χ.

    Measures closed timelike curves in the effective metric reconstructed
    from the dispersion relation.

    Args:
        model: Model instance (must be LatticeFieldModel)
        results: Simulation results containing effective_metric_data

    Returns:
        Dictionary with:
            - chi: Loop functional value (negative indicates timelike loops)
            - flow_velocity: Flow velocity parameter
            - c_eff: Effective light speed
    """
    # Only compute for lattice models with effective metric
    if not hasattr(model, "effective_metric_data"):
        return {
            "chi": None,
            "flow_velocity": None,
            "c_eff": None,
            "note": "Chi not applicable for this model type",
        }

    metric_data = results.get("effective_metric_data", model.effective_metric_data)

    flow_vel = metric_data["flow_velocity"]
    c_eff = metric_data["c_eff"]

    # Compute chi as constrained loop functional
    # For a tilted light cone with flow velocity v:
    # χ ~ (v² - c²) where negative indicates timelike loops

    # Normalize velocities
    chi = flow_vel**2 - c_eff**2

    # Additional geometric factor from lattice discretization
    # Include dispersion effect
    dispersion_factor = 1.0 + metric_data["dispersion_deviation"]
    chi = chi * dispersion_factor

    return {
        "chi": float(chi),
        "flow_velocity": float(flow_vel),
        "c_eff": float(c_eff),
        "dispersion_deviation": float(metric_data["dispersion_deviation"]),
    }


def compute_chi_statistics(chi_results: list) -> Dict[str, Any]:
    """
    Compute statistics over multiple chi measurements.

    Args:
        chi_results: List of chi computation results

    Returns:
        Dictionary with aggregate statistics
    """
    chi_values = [r["chi"] for r in chi_results if r["chi"] is not None]

    if not chi_values:
        return {
            "chi_mean": None,
            "chi_std": None,
            "chi_min": None,
            "chi_max": None,
            "n_negative": 0,
            "n_valid": 0,
        }

    chi_array = np.array(chi_values)

    return {
        "chi_mean": float(np.mean(chi_array)),
        "chi_std": float(np.std(chi_array)),
        "chi_min": float(np.min(chi_array)),
        "chi_max": float(np.max(chi_array)),
        "n_negative": int(np.sum(chi_array < 0)),
        "n_valid": len(chi_values),
    }

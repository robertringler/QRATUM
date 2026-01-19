"""Anti-holographic sensitivity analysis.

Tests hypothesis: χ is more sensitive to bulk parameters than boundary parameters.
"""

from typing import Any, Dict, List

import numpy as np


def compute_finite_difference_sensitivity(
    chi_results: List[Dict[str, Any]], parameter_name: str, parameter_values: List[float]
) -> Dict[str, Any]:
    """
    Compute finite-difference sensitivity of χ to a parameter.

    Sensitivity = dχ/dθ ≈ Δχ/Δθ

    Args:
        chi_results: List of chi computation results
        parameter_name: Name of parameter being varied
        parameter_values: Values of the parameter

    Returns:
        Dictionary with sensitivity statistics
    """
    if len(chi_results) < 2:
        return {
            "parameter": parameter_name,
            "sensitivity_mean": None,
            "sensitivity_std": None,
            "n_points": len(chi_results),
        }

    # Extract chi values
    chi_values = [r.get("chi", 0.0) for r in chi_results if r.get("chi") is not None]

    if len(chi_values) < 2 or len(parameter_values) < 2:
        return {
            "parameter": parameter_name,
            "sensitivity_mean": None,
            "sensitivity_std": None,
            "n_points": len(chi_values),
        }

    # Compute finite differences
    sensitivities = []
    for i in range(len(chi_values) - 1):
        dchi = chi_values[i + 1] - chi_values[i]
        dtheta = parameter_values[i + 1] - parameter_values[i]

        if abs(dtheta) > 1e-10:
            sensitivity = abs(dchi / dtheta)
            sensitivities.append(sensitivity)

    if not sensitivities:
        return {
            "parameter": parameter_name,
            "sensitivity_mean": None,
            "sensitivity_std": None,
            "n_points": 0,
        }

    sens_array = np.array(sensitivities)

    return {
        "parameter": parameter_name,
        "sensitivity_mean": float(np.mean(sens_array)),
        "sensitivity_std": float(np.std(sens_array)),
        "sensitivity_median": float(np.median(sens_array)),
        "n_points": len(sensitivities),
    }


def bootstrap_sensitivity_confidence(
    chi_values: np.ndarray,
    parameter_values: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for sensitivity.

    Args:
        chi_values: Array of chi values
        parameter_values: Array of parameter values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Dictionary with confidence intervals
    """
    if len(chi_values) < 3:
        return {
            "lower_bound": None,
            "upper_bound": None,
            "mean": None,
        }

    n_points = len(chi_values)
    sensitivities_boot = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(n_points, size=n_points, replace=True)
        chi_boot = chi_values[indices]
        param_boot = parameter_values[indices]

        # Sort by parameter value
        sort_idx = np.argsort(param_boot)
        chi_boot = chi_boot[sort_idx]
        param_boot = param_boot[sort_idx]

        # Compute sensitivity
        sensitivities = []
        for i in range(len(chi_boot) - 1):
            dchi = chi_boot[i + 1] - chi_boot[i]
            dtheta = param_boot[i + 1] - param_boot[i]

            if abs(dtheta) > 1e-10:
                sensitivities.append(abs(dchi / dtheta))

        if sensitivities:
            sensitivities_boot.append(np.mean(sensitivities))

    if not sensitivities_boot:
        return {
            "lower_bound": None,
            "upper_bound": None,
            "mean": None,
        }

    sens_boot_array = np.array(sensitivities_boot)
    alpha = 1 - confidence_level

    return {
        "lower_bound": float(np.percentile(sens_boot_array, 100 * alpha / 2)),
        "upper_bound": float(np.percentile(sens_boot_array, 100 * (1 - alpha / 2))),
        "mean": float(np.mean(sens_boot_array)),
    }


def analyze_bulk_vs_boundary_sensitivity(results_df) -> Dict[str, Any]:
    """
    Analyze sensitivity of χ to bulk vs boundary parameters.

    Bulk parameters: cutoff_scale, dispersion_strength, dissipation_rate
    Boundary parameters: n_sites (system_size)

    Args:
        results_df: DataFrame with simulation results

    Returns:
        Dictionary with comparative sensitivity analysis
    """
    import pandas as pd

    if not isinstance(results_df, pd.DataFrame):
        return {
            "note": "No results dataframe available",
            "bulk_sensitivity": None,
            "boundary_sensitivity": None,
        }

    # Filter to lattice model results with valid chi
    lattice_results = results_df[
        (results_df["model_type"] == "lattice") & (results_df["chi"].notna())
    ]

    if len(lattice_results) < 2:
        return {
            "note": "Insufficient lattice results for sensitivity analysis",
            "bulk_sensitivity": None,
            "boundary_sensitivity": None,
        }

    sensitivities = {}

    # Bulk parameters
    bulk_params = ["cutoff_scale", "dispersion_strength", "dissipation_rate"]
    for param in bulk_params:
        if param in lattice_results.columns:
            # Group by parameter and compute average sensitivity
            param_groups = lattice_results.groupby(param)["chi"].apply(list)
            if len(param_groups) >= 2:
                chi_vals = []
                param_vals = []
                for pval, chi_list in param_groups.items():
                    chi_vals.append(np.mean(chi_list))
                    param_vals.append(pval)

                sens = compute_finite_difference_sensitivity(
                    [{"chi": c} for c in chi_vals], param, param_vals
                )
                sensitivities[param] = sens

    # Boundary parameter
    if "n_sites" in lattice_results.columns:
        param_groups = lattice_results.groupby("n_sites")["chi"].apply(list)
        if len(param_groups) >= 2:
            chi_vals = []
            param_vals = []
            for pval, chi_list in param_groups.items():
                chi_vals.append(np.mean(chi_list))
                param_vals.append(pval)

            sens = compute_finite_difference_sensitivity(
                [{"chi": c} for c in chi_vals], "n_sites", param_vals
            )
            sensitivities["n_sites"] = sens

    # Compute aggregate statistics
    bulk_sens_values = [
        s["sensitivity_mean"]
        for s in sensitivities.values()
        if s.get("parameter") in bulk_params and s.get("sensitivity_mean") is not None
    ]

    boundary_sens_values = [
        s["sensitivity_mean"]
        for s in sensitivities.values()
        if s.get("parameter") == "n_sites" and s.get("sensitivity_mean") is not None
    ]

    return {
        "individual_sensitivities": sensitivities,
        "bulk_sensitivity_mean": float(np.mean(bulk_sens_values)) if bulk_sens_values else None,
        "bulk_sensitivity_std": (
            float(np.std(bulk_sens_values)) if len(bulk_sens_values) > 1 else None
        ),
        "boundary_sensitivity_mean": (
            float(np.mean(boundary_sens_values)) if boundary_sens_values else None
        ),
        "hypothesis_supported": (
            bool(np.mean(bulk_sens_values) > np.mean(boundary_sens_values))
            if bulk_sens_values and boundary_sens_values
            else None
        ),
    }

"""Lab-time signaling diagnostic (S_lab).

S_lab(θ) = sup_{O,I∈A} | ∂P(O_{t1}) / ∂I_{t2} | for t1 < t2
"""

from typing import Any, Dict

import numpy as np


def compute_s_lab(model, n_samples: int = 100) -> Dict[str, Any]:
    """
    Compute lab-time signaling diagnostic S_lab.
    
    Measures whether future interventions at t2 can signal backward to
    observables at t1 < t2.
    
    Args:
        model: Model instance (must have run_twin_simulations method)
        n_samples: Number of sample runs for statistical estimate
        
    Returns:
        Dictionary with:
            - s_lab: Point estimate of signaling strength
            - s_lab_stderr: Standard error
            - observables_I0: Array of observables with I=0
            - observables_I1: Array of observables with I=1
            - p_value: Statistical significance
    """
    obs_I0_list = []
    obs_I1_list = []

    # Run multiple samples to estimate distribution
    original_seed = model.seed

    for i in range(n_samples):
        # Use different seed for each sample
        model.seed = original_seed + i
        model.rng = np.random.RandomState(model.seed)

        # Run twin simulations
        results_I0, results_I1 = model.run_twin_simulations()

        # Extract observables at t1 (causality audit ensures no future data access)
        obs_I0 = model.get_observable_t1(results_I0)
        obs_I1 = model.get_observable_t1(results_I1)

        obs_I0_list.append(obs_I0)
        obs_I1_list.append(obs_I1)

    # Restore original seed
    model.seed = original_seed

    # Convert to arrays
    obs_I0 = np.array(obs_I0_list)
    obs_I1 = np.array(obs_I1_list)

    # Compute S_lab as mean absolute difference
    differences = np.abs(obs_I1 - obs_I0)
    s_lab = np.mean(differences)
    s_lab_stderr = np.std(differences) / np.sqrt(n_samples)

    # Compute p-value (two-sample t-test equivalent)
    # Under null hypothesis (no signaling), differences should be zero
    if s_lab_stderr > 0:
        t_statistic = s_lab / s_lab_stderr
        # Approximate p-value (assuming normal distribution)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(t_statistic)))
    else:
        p_value = 1.0

    return {
        's_lab': float(s_lab),
        's_lab_stderr': float(s_lab_stderr),
        'observables_I0': obs_I0.tolist(),
        'observables_I1': obs_I1.tolist(),
        'n_samples': n_samples,
        'p_value': float(p_value),
    }


def check_s_lab_significance(s_lab_result: Dict[str, Any],
                             threshold_sigma: float = 5.0) -> bool:
    """
    Check if S_lab is statistically significant.
    
    Args:
        s_lab_result: Result from compute_s_lab
        threshold_sigma: Significance threshold in standard deviations
        
    Returns:
        True if S_lab exceeds threshold_sigma * stderr
    """
    s_lab = s_lab_result['s_lab']
    stderr = s_lab_result['s_lab_stderr']

    if stderr == 0:
        return False

    significance = s_lab / stderr
    return significance > threshold_sigma

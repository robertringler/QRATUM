"""Phase 3 acceptance test: closed-loop active experiment design.

The directive's differentiator: given the current posterior, propose the next
experiment that maximizes expected information gain. Acceptance criterion - the
active-design loop reaches a tighter posterior in measurably fewer experiments
than random experiment selection.
"""

from __future__ import annotations

import numpy as np

from xenon.learning.active_design import (
    CandidateExperiment,
    run_active_design_loop,
    select_next_experiment,
)
from xenon.learning.forward_inference import MechanismModel, metropolis_infer, Observation, simulate_meanfield

SPECIES = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "C")]
TRUE_RATES = [0.8, 0.3]


def _candidate_pool():
    """A pool spanning informative and uninformative experiments.

    Short times with all mass on A barely move B/C (low information for k2);
    longer times and pre-loaded B are informative about both rates.
    """
    pool = []
    for init in (
        {"A": 100.0, "B": 0.0, "C": 0.0},
        {"A": 0.0, "B": 100.0, "C": 0.0},
        {"A": 50.0, "B": 50.0, "C": 0.0},
    ):
        for t in (0.1, 0.5, 1.0, 3.0, 6.0):
            pool.append(CandidateExperiment(initial=init, time=t, observed_species=["A", "B", "C"]))
    return pool


def test_active_design_beats_random_on_average():
    model = MechanismModel(SPECIES, EDGES)
    candidates = _candidate_pool()
    n_experiments = 5

    # Average over several seeds to compare strategies robustly.
    active_final, random_final = [], []
    for seed in range(5):
        active = run_active_design_loop(
            model, TRUE_RATES, candidates, n_experiments, strategy="active", seed=seed
        )
        random = run_active_design_loop(
            model, TRUE_RATES, candidates, n_experiments, strategy="random", seed=seed
        )
        active_final.append(active.posterior_widths[-1])
        random_final.append(random.posterior_widths[-1])

    mean_active = float(np.mean(active_final))
    mean_random = float(np.mean(random_final))

    # Active design should reach a tighter (smaller) posterior than random.
    assert mean_active < mean_random, (
        f"active posterior width {mean_active:.4f} not < random {mean_random:.4f}"
    )


def test_select_next_experiment_prefers_higher_variance():
    """The selector must pick the candidate with greatest predictive variance."""
    model = MechanismModel(SPECIES, EDGES)
    # A spread of posterior rate samples so predictions disagree.
    rng = np.random.default_rng(0)
    posterior = np.column_stack(
        [10 ** rng.uniform(-1.0, 0.5, size=200), 10 ** rng.uniform(-1.0, 0.5, size=200)]
    )
    candidates = _candidate_pool()
    idx, score = select_next_experiment(model, posterior, candidates, noise=2.0)
    assert 0 <= idx < len(candidates)
    assert score >= 0.0
    # The very-short-time, A-only experiment is among the least informative.
    uninformative = CandidateExperiment({"A": 100.0, "B": 0.0, "C": 0.0}, 0.1, ["A", "B", "C"])
    from xenon.learning.active_design import expected_information_gain

    assert score >= expected_information_gain(model, posterior, uninformative, noise=2.0)

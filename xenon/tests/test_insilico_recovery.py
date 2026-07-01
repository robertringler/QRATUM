"""Closed-loop in-silico recovery tests for XENON inference (Tier-0 validation).

These tests verify that, with the corrected simulator and likelihood, XENON
recovers a known ground-truth mechanism from data the mechanism itself generated
— and that it does so for the right (identifiable) reason.
"""

from __future__ import annotations

import numpy as np

from xenon.validation.insilico_recovery import (
    make_pathway_mechanism,
    make_two_state_mechanism,
    predict_steady_state,
    run_perturbation_recovery,
    run_recovery,
)

PROTEIN = "EGFR"


def test_recovers_true_K_and_accumulates_evidence():
    """Posterior concentrates on the true-K mechanism and grows with data (F4)."""

    truth = make_two_state_mechanism("M_star", PROTEIN, 2.0, 1.0)
    specs = [("C_K05", 1.0, 2.0), ("C_K1", 1.0, 1.0), ("C_K2", 2.0, 1.0),
             ("C_K4", 4.0, 1.0)]
    candidates = [make_two_state_mechanism(n, PROTEIN, kf, kr) for n, kf, kr in specs]

    result = run_recovery(
        truth=truth, candidates=candidates, protein=PROTEIN,
        n_experiments=20, measurement_noise_nM=5.0, seed=42, replicates=16,
    )

    # The true mechanism (K=2) is recovered with dominant posterior.
    assert result.recovered_name == "C_K2"
    assert result.final_posterior["C_K2"] > 0.9

    # Evidence accumulates: posterior on the truth is higher after all
    # experiments than after the first.
    curve = [step["C_K2"] for step in result.posterior_trajectory]
    assert curve[-1] > curve[0]

    # Wrong-K decoys are strongly suppressed.
    for name in ("C_K05", "C_K1", "C_K4"):
        assert result.final_posterior[name] < 0.05


def test_equal_K_is_non_identifiable_from_concentration():
    """Equal-K mechanisms have predictions that agree within Monte-Carlo error."""

    c_k2 = make_two_state_mechanism("C_K2", PROTEIN, 2.0, 1.0)
    c_k2_alt = make_two_state_mechanism("C_K2_alt", PROTEIN, 4.0, 2.0)

    rng = np.random.default_rng(42)
    p1 = predict_steady_state(c_k2, rng, replicates=64, t_max=10.0)
    p2 = predict_steady_state(c_k2_alt, rng, replicates=64, t_max=10.0)

    active = f"{PROTEIN}_active"
    mean_diff = abs(p1[active][0] - p2[active][0])
    mc_se = (p1[active][1] ** 2 + p2[active][1] ** 2) ** 0.5

    # Predictions agree within ~2 Monte-Carlo standard errors -> not separable
    # from a steady-state concentration measurement.
    assert mean_diff <= 2.0 * mc_se + 1.0


def test_perturbation_recovery_recovers_topology():
    """Perturbation channel recovers the true topology and rejects wrong ones (F3)."""

    src, tgt = f"{PROTEIN}_inactive", f"{PROTEIN}_active"
    truth = make_pathway_mechanism("M_star", PROTEIN, [("inactive", "active")])
    candidates = [
        make_pathway_mechanism("C_direct", PROTEIN, [("inactive", "active")]),
        make_pathway_mechanism("C_indirect", PROTEIN, [("inactive", "mid"), ("mid", "active")]),
        make_pathway_mechanism("C_nopath", PROTEIN, [("active", "inactive")]),
    ]

    result = run_perturbation_recovery(
        truth, candidates, src, tgt, n_experiments=16, response_noise=0.15, seed=42
    )

    # True topology recovered; no-path mechanism rejected; evidence accumulates.
    assert result.recovered_name == "C_direct"
    assert result.final_posterior["C_direct"] > 0.9
    assert result.final_posterior["C_nopath"] < 1e-3
    curve = [step["C_direct"] for step in result.posterior_trajectory]
    assert curve[-1] > curve[0]


def test_recovery_is_deterministic():
    """Same seed -> identical recovery outcome."""

    def go():
        truth = make_two_state_mechanism("M_star", PROTEIN, 2.0, 1.0)
        candidates = [
            make_two_state_mechanism("C_K1", PROTEIN, 1.0, 1.0),
            make_two_state_mechanism("C_K2", PROTEIN, 2.0, 1.0),
        ]
        return run_recovery(truth, candidates, PROTEIN, n_experiments=10, seed=7).final_posterior

    a, b = go(), go()
    assert a == b

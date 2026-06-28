#!/usr/bin/env python3
"""In-silico recovery campaign: can XENON recover a known ground-truth mechanism?

Two scenarios, both with ground truth M* = reversible EGFR activation, K=2
(k_f=2, k_r=1), total 100 nM, observed as noisy steady-state concentrations.

Scenario A — RECOVERY + EVIDENCE ACCUMULATION
  Candidates span the identifiable quantity K in {0.5, 1, 1.5, 2(true), 2.5, 4}.
  Validates: (1) the posterior concentrates on the true K=2 mechanism and
  eliminates wrong-K decoys; (2) posterior on the truth grows with the number of
  experiments (i.e. evidence accumulates — the direct test of fix F4).

Scenario B — IDENTIFIABILITY (honest)
  Candidates C_K2 (k_f=2,k_r=1) and C_K2_alt (k_f=4,k_r=2): SAME K, different
  absolute rates. A two-state chain's stationary distribution depends only on
  K, so these are analytically indistinguishable from steady-state concentration.
  The harness reports that their predicted observables agree within Monte-Carlo
  error, i.e. any posterior split is a simulation-resolution artifact, not real
  discrimination — separating them requires kinetic/time-course data (motivates
  Phase 5 identifiability analysis and Phase 6 optimal experiment design).

Output: xenon_campaigns/recovery/recovery_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xenon.validation.insilico_recovery import (
    make_two_state_mechanism,
    predict_steady_state,
    run_recovery,
)

PROTEIN = "EGFR"
SEED = 42


def scenario_a() -> dict:
    truth = make_two_state_mechanism("M_star", PROTEIN, 2.0, 1.0)
    specs = [("C_K05", 1.0, 2.0), ("C_K1", 1.0, 1.0), ("C_K1_5", 1.5, 1.0),
             ("C_K2", 2.0, 1.0), ("C_K2_5", 2.5, 1.0), ("C_K4", 4.0, 1.0)]
    candidates = [make_two_state_mechanism(n, PROTEIN, kf, kr) for n, kf, kr in specs]

    result = run_recovery(
        truth=truth, candidates=candidates, protein=PROTEIN,
        n_experiments=20, measurement_noise_nM=5.0, seed=SEED, replicates=16,
    )
    truth_posterior_curve = [round(step["C_K2"], 4) for step in result.posterior_trajectory]
    return {
        "truth": {"name": "M_star", "K": 2.0},
        "candidate_K": result.candidate_K,
        "truth_steady_state_nM": result.truth_steady_state,
        "final_posterior": {k: round(v, 6) for k, v in result.final_posterior.items()},
        "recovered_name": result.recovered_name,
        "truth_posterior_over_experiments": truth_posterior_curve,
        "verdict": {
            "recovers_true_mechanism": result.recovered_name == "C_K2",
            "final_posterior_on_truth": round(result.final_posterior["C_K2"], 4),
            "evidence_accumulates": truth_posterior_curve[-1] > truth_posterior_curve[0],
        },
    }


def scenario_b() -> dict:
    truth = make_two_state_mechanism("M_star", PROTEIN, 2.0, 1.0)
    c_k2 = make_two_state_mechanism("C_K2", PROTEIN, 2.0, 1.0)
    c_k2_alt = make_two_state_mechanism("C_K2_alt", PROTEIN, 4.0, 2.0)

    # Compare the two candidates' predicted observables directly (high replicates,
    # long horizon) to show they agree within Monte-Carlo error.
    rng = np.random.default_rng(SEED)
    p1 = predict_steady_state(c_k2, rng, replicates=64, t_max=10.0)
    p2 = predict_steady_state(c_k2_alt, rng, replicates=64, t_max=10.0)
    active = f"{PROTEIN}_active"
    mean_diff = abs(p1[active][0] - p2[active][0])
    mc_se = (p1[active][1] ** 2 + p2[active][1] ** 2) ** 0.5

    result = run_recovery(
        truth=truth, candidates=[c_k2, c_k2_alt], protein=PROTEIN,
        n_experiments=20, measurement_noise_nM=5.0, seed=SEED, replicates=64,
    )
    return {
        "predicted_active_nM": {"C_K2": round(p1[active][0], 3), "C_K2_alt": round(p2[active][0], 3)},
        "prediction_mean_diff_nM": round(mean_diff, 3),
        "prediction_mc_se_nM": round(mc_se, 3),
        "predictions_agree_within_mc_error": mean_diff <= 2.0 * mc_se,
        "final_posterior": {k: round(v, 4) for k, v in result.final_posterior.items()},
        "interpretation": (
            "C_K2 and C_K2_alt have identical stationary distributions (depend only "
            "on K); their predicted observables agree within Monte-Carlo error, so any "
            "posterior split is a simulation-resolution artifact, NOT identifiability. "
            "Separating equal-K mechanisms requires kinetic/time-course data."
        ),
    }


def main() -> int:
    record = {
        "scenario_meta": {"protein": PROTEIN, "seed": SEED,
                          "ground_truth": {"k_f": 2.0, "k_r": 1.0, "K": 2.0}},
        "scenario_A_recovery": scenario_a(),
        "scenario_B_identifiability": scenario_b(),
    }
    out_dir = Path("xenon_campaigns") / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recovery_results.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))

    a = record["scenario_A_recovery"]
    b = record["scenario_B_identifiability"]
    print(f"Wrote {out_path}")
    print("[A] recovered:", a["recovered_name"],
          "| posterior on truth:", a["verdict"]["final_posterior_on_truth"],
          "| accumulates:", a["verdict"]["evidence_accumulates"])
    print("    truth posterior curve:", a["truth_posterior_over_experiments"])
    print("[B] pred active C_K2 vs C_K2_alt:", b["predicted_active_nM"],
          "| diff", b["prediction_mean_diff_nM"], "nM vs 2*MC_SE", round(2 * b["prediction_mc_se_nM"], 3),
          "| within MC error:", b["predictions_agree_within_mc_error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

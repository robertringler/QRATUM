"""In-silico closed-loop recovery harness for XENON inference.

Purpose
-------
Replace the previously mechanism-independent mock experiment executor (audit
finding F7) with a *generative* ground truth, so that inference can be validated
rather than merely reproduced:

1. Choose a known ground-truth mechanism M* with fixed kinetic parameters.
2. Generate experimental observations by simulating M* (corrected SSA) and adding
   measurement noise.
3. Present a candidate set of mechanisms (one or more matching M*'s identifiable
   quantity, plus decoys).
4. Run sequential Bayesian updating on the M*-derived observations.
5. Verify the posterior concentrates on the mechanism(s) consistent with M* and
   that evidence accumulates with the number of experiments.

This exercises the Tier-0 fixes end to end:
  * F1 (reverse reactions) — steady states are correct, so the observable is real.
  * F4 (joint likelihood)  — posterior on the true mechanism grows with data.
  * F6 (replicated observable + MC SE) — simulation noise enters the noise model.

Everything is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xenon.core.mechanism import BioMechanism, MolecularState, Transition
from xenon.learning.bayesian_updater import BayesianUpdater, ExperimentResult
from xenon.simulation.gillespie import GillespieSimulator

ACTIVE = "active"
INACTIVE = "inactive"


def make_two_state_mechanism(
    name: str,
    protein: str,
    k_forward: float,
    k_reverse: float,
) -> BioMechanism:
    """Build a reversible two-state mechanism ``inactive <-> active``.

    The steady-state split is governed by ``K = k_forward / k_reverse`` (the only
    quantity identifiable from a steady-state concentration measurement).
    """

    mech = BioMechanism(name)
    mech.add_state(
        MolecularState(name=f"{protein}_{INACTIVE}", molecule=protein, concentration=50.0,
                       free_energy=-10.0)
    )
    mech.add_state(
        MolecularState(name=f"{protein}_{ACTIVE}", molecule=protein, concentration=50.0,
                       free_energy=-12.0)
    )
    mech.add_transition(
        Transition(
            source=f"{protein}_{INACTIVE}",
            target=f"{protein}_{ACTIVE}",
            rate_constant=k_forward,
            reversible=True,
            reverse_rate=k_reverse,
        )
    )
    return mech


def predict_steady_state(
    mechanism: BioMechanism,
    rng: np.random.Generator,
    replicates: int = 8,
    t_max: float = 5.0,
    total_conc: float = 100.0,
) -> dict[str, tuple[float, float]]:
    """Estimate steady-state concentrations by replicated SSA.

    Returns ``{state_name: (mean_nM, mc_standard_error)}``. Initial mass is
    placed entirely in the inactive state so the relaxation to the equilibrium
    split is meaningful.
    """

    state_names = list(mechanism._states)
    initial = dict.fromkeys(state_names, 0.0)
    # All mass starts inactive.
    inactive = next((s for s in state_names if s.endswith(INACTIVE)), state_names[0])
    initial[inactive] = total_conc

    samples: dict[str, list[float]] = {s: [] for s in state_names}
    sim = GillespieSimulator(mechanism, volume=1e-15)
    for _ in range(replicates):
        seed = int(rng.integers(2**31))
        _, traj = sim.run(
            t_max=t_max, initial_state=dict(initial), seed=seed, record_interval=t_max / 20.0
        )
        for s in state_names:
            series = traj.get(s)
            if series:
                samples[s].append(series[-1])

    out: dict[str, tuple[float, float]] = {}
    for s, vals in samples.items():
        if vals:
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[s] = (mean, se)
    return out


def _apply_prediction(mechanism: BioMechanism, prediction: dict[str, tuple[float, float]]) -> None:
    """Write a predicted observable (mean + MC SE) onto a mechanism's states."""

    for state_name, (mean, se) in prediction.items():
        if state_name in mechanism._states:
            state = mechanism._states[state_name]
            state.concentration = mean
            state.properties["conc_mc_se"] = se


@dataclass
class RecoveryResult:
    """Outcome of an in-silico recovery run."""

    truth_name: str
    candidate_names: list[str]
    truth_steady_state: dict[str, float]
    posterior_trajectory: list[dict[str, float]]  # per experiment
    final_posterior: dict[str, float]
    recovered_name: str
    truth_identifiable_K: float
    candidate_K: dict[str, float] = field(default_factory=dict)


def run_recovery(
    truth: BioMechanism,
    candidates: list[BioMechanism],
    protein: str,
    n_experiments: int = 12,
    measurement_noise_nM: float = 3.0,
    seed: int = 42,
    replicates: int = 8,
) -> RecoveryResult:
    """Run closed-loop recovery and return the posterior trajectory.

    Each experiment is a noisy steady-state concentration measurement generated
    from ``truth``. Candidate predictions are computed once (deterministically)
    and reused across experiments; only the (noisy) observations change.
    """

    rng = np.random.default_rng(seed)

    # Ground-truth steady state (the generative model).
    truth_ss = predict_steady_state(truth, rng, replicates=replicates)
    truth_means = {s: m for s, (m, _) in truth_ss.items()}

    # Candidate predictions (each candidate's own steady state under the fixed SSA).
    candidate_predictions = {}
    for cand in candidates:
        candidate_predictions[cand.name] = predict_steady_state(cand, rng, replicates=replicates)
        _apply_prediction(cand, candidate_predictions[cand.name])

    # Uniform prior over candidates.
    for cand in candidates:
        cand.posterior = 1.0 / len(candidates)

    updater = BayesianUpdater()
    active = f"{protein}_{ACTIVE}"
    inactive = f"{protein}_{INACTIVE}"

    trajectory: list[dict[str, float]] = []
    for _ in range(n_experiments):
        # Generate a noisy concentration observation from the truth.
        obs = {
            inactive: truth_means.get(inactive, 0.0) + rng.normal(0.0, measurement_noise_nM),
            active: truth_means.get(active, 0.0) + rng.normal(0.0, measurement_noise_nM),
        }
        experiment = ExperimentResult(
            experiment_type="concentration",
            observations=obs,
            uncertainties={inactive: measurement_noise_nM, active: measurement_noise_nM},
            conditions={"temperature": 310.0},
        )
        updater.update_mechanisms(candidates, experiment)
        trajectory.append({c.name: c.posterior for c in candidates})

    final = {c.name: c.posterior for c in candidates}
    recovered = max(final, key=final.get)

    def K_of(m: BioMechanism) -> float:
        t = m._transitions[0]
        return t.rate_constant / (t.reverse_rate or np.nan)

    return RecoveryResult(
        truth_name=truth.name,
        candidate_names=[c.name for c in candidates],
        truth_steady_state={s: round(m, 3) for s, m in truth_means.items()},
        posterior_trajectory=trajectory,
        final_posterior=final,
        recovered_name=recovered,
        truth_identifiable_K=K_of(truth),
        candidate_K={c.name: K_of(c) for c in candidates},
    )

"""Closed-loop optimal experiment design (the directive's differentiator).

Given a posterior over a mechanism's rate constants, choose the next experiment
(initial conditions + observation time, from a candidate pool) that is expected
to be most informative, then fold its result back into the posterior and repeat.

Information criterion
---------------------
For a Gaussian observation model the experiment that most reduces parameter
uncertainty is, to first order, the one whose model prediction is *most uncertain*
under the current posterior (Bayesian active learning by disagreement / maximum
predictive variance - a standard tractable surrogate for expected information
gain). We score a candidate by the total posterior-predictive variance of its
observed species, normalized by measurement noise.

This module reuses the forward model and the Metropolis sampler from
``forward_inference``; it adds no new modelling assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .forward_inference import (
    MechanismModel,
    Observation,
    metropolis_infer,
)


@dataclass
class CandidateExperiment:
    """A proposable experiment: initial conditions, observation time, readouts."""

    initial: dict[str, float]
    time: float
    observed_species: list[str]


def expected_information_gain(
    model: MechanismModel,
    posterior_rates: np.ndarray,  # (n_samples, n_params)
    candidate: CandidateExperiment,
    noise: float = 1.0,
) -> float:
    """Posterior-predictive-variance score (EIG surrogate) for a candidate.

    Higher means the candidate's outcome is currently more uncertain and thus
    more informative to run.
    """
    preds = np.array(
        [
            [model.predict(rates, candidate.initial, candidate.time)[s] for s in candidate.observed_species]
            for rates in posterior_rates
        ]
    )  # (n_samples, n_obs)
    predictive_var = preds.var(axis=0)  # per observed species
    return float(np.sum(predictive_var) / (noise**2))


def select_next_experiment(
    model: MechanismModel,
    posterior_rates: np.ndarray,
    candidates: Sequence[CandidateExperiment],
    noise: float = 1.0,
) -> tuple[int, float]:
    """Return (index, score) of the most informative candidate experiment."""
    scores = [
        expected_information_gain(model, posterior_rates, c, noise) for c in candidates
    ]
    best = int(np.argmax(scores))
    return best, scores[best]


def _run_experiment(
    model: MechanismModel,
    true_rates: Sequence[float],
    candidate: CandidateExperiment,
    noise: float,
    rng: np.random.Generator,
) -> Observation:
    """Simulate the ground-truth outcome of a candidate experiment (with noise)."""
    truth = model.predict(true_rates, candidate.initial, candidate.time)
    observed = {s: float(truth[s] + rng.normal(0.0, noise)) for s in candidate.observed_species}
    uncertainties = {s: noise for s in candidate.observed_species}
    return Observation(
        initial=candidate.initial, time=candidate.time, observed=observed, uncertainties=uncertainties
    )


@dataclass
class ActiveDesignTrace:
    """Per-iteration record of a closed-loop design run."""

    posterior_widths: list[float]
    chosen: list[int]


def run_active_design_loop(
    model: MechanismModel,
    true_rates: Sequence[float],
    candidates: Sequence[CandidateExperiment],
    n_experiments: int,
    noise: float = 2.0,
    strategy: str = "active",
    n_mcmc: int = 800,
    burn_in: int = 300,
    seed: Optional[int] = None,
) -> ActiveDesignTrace:
    """Run the simulate -> infer -> design -> repeat loop.

    Args:
        model: the (known) topology being calibrated.
        true_rates: ground-truth rates used to synthesise experiment outcomes.
        candidates: pool of proposable experiments.
        n_experiments: number of experiments to run.
        strategy: "active" (max expected information gain) or "random".

    Returns:
        ActiveDesignTrace with the posterior width (mean parameter std in log10
        space) after each experiment - the smaller, the more confident.
    """
    rng = np.random.default_rng(seed)
    data: list[Observation] = []
    widths: list[float] = []
    chosen: list[int] = []

    # Seed with one random experiment so the first posterior is defined.
    first = int(rng.integers(len(candidates)))
    data.append(_run_experiment(model, true_rates, candidates[first], noise, rng))
    chosen.append(first)

    for _ in range(n_experiments):
        result = metropolis_infer(
            model, data, n_samples=n_mcmc, burn_in=burn_in, seed=int(rng.integers(1 << 31))
        )
        # Posterior width = mean over params of std of log10(rate) samples.
        widths.append(float(np.mean(np.std(np.log10(result.samples), axis=0))))

        if strategy == "active":
            idx, _ = select_next_experiment(model, result.samples, candidates, noise)
        else:
            idx = int(rng.integers(len(candidates)))
        chosen.append(idx)
        data.append(_run_experiment(model, true_rates, candidates[idx], noise, rng))

    # Final posterior width after the last experiment.
    result = metropolis_infer(
        model, data, n_samples=n_mcmc, burn_in=burn_in, seed=int(rng.integers(1 << 31))
    )
    widths.append(float(np.mean(np.std(np.log10(result.samples), axis=0))))

    return ActiveDesignTrace(posterior_widths=widths, chosen=chosen)


__all__ = [
    "CandidateExperiment",
    "ActiveDesignTrace",
    "expected_information_gain",
    "select_next_experiment",
    "run_active_design_loop",
]

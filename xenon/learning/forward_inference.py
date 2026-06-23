"""Forward-model Bayesian inference for biochemical mechanisms.

This module is the substance behind XENON's mechanism-inference vertical. It
replaces the previous "score against the stored concentration" likelihood with a
real one: given a candidate mechanism (a topology + kinetic rate constants) and
an experiment (initial conditions + observation time), it *forward-simulates* the
mechanism and scores the predicted observables against the data.

For the unimolecular reaction networks XENON models (A -> B with propensity
k*[A]), the mean of the chemical master equation obeys the linear ODE

    dC/dt = Q C,      C(t) = expm(Q t) C(0)

where ``Q`` is the reaction generator built from the network's edges and rates.
This mean-field prediction is exact for the first moment, deterministic, and fast
(a single matrix exponential), which makes the likelihood tractable and the
inference reliable. The stochastic Gillespie path remains available in
``xenon.simulation.gillespie`` for discrete-count regimes.

On top of the likelihood this module provides:
  * ``MechanismModel`` - a topology (species + directed edges) whose rates are
    the free parameters of inference;
  * ``metropolis_infer`` - random-walk Metropolis MCMC over log10(rates) using the
    analytic Gaussian likelihood (joint parameter inference);
  * ``marginal_likelihood`` / ``model_selection`` - Monte-Carlo evidence over the
    prior for Bayesian model (topology) selection.

These are deliberately self-contained (numpy + scipy.linalg.expm); they can be
swapped for pyABC / PyMC backends without changing the model or likelihood API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm

from ..core.mechanism import BioMechanism

Edge = tuple[str, str]


# --------------------------------------------------------------------------- #
# Forward model (mean-field)
# --------------------------------------------------------------------------- #
def _generator_matrix(
    species: Sequence[str], edges: Sequence[Edge], rates: Sequence[float]
) -> np.ndarray:
    """Build the linear reaction generator Q for ``dC/dt = Q C``.

    For each edge (source -> target) with rate k: mass leaves the source
    (``Q[src, src] -= k``) and arrives at the target (``Q[tgt, src] += k``).
    """
    index = {name: i for i, name in enumerate(species)}
    n = len(species)
    q = np.zeros((n, n), dtype=float)
    for (source, target), k in zip(edges, rates):
        i, j = index[source], index[target]
        q[i, i] -= k
        q[j, i] += k
    return q


def simulate_meanfield(
    species: Sequence[str],
    edges: Sequence[Edge],
    rates: Sequence[float],
    initial: dict[str, float],
    t: float,
) -> dict[str, float]:
    """Predict mean concentrations at time ``t`` via ``expm(Q t) C0``."""
    q = _generator_matrix(species, edges, rates)
    c0 = np.array([float(initial.get(name, 0.0)) for name in species], dtype=float)
    ct = expm(q * float(t)) @ c0
    return {name: float(ct[i]) for i, name in enumerate(species)}


def mechanism_edges(mechanism: BioMechanism) -> list[Edge]:
    """Directed edges (source, target) of a mechanism, in transition order."""
    return [(t.source, t.target) for t in mechanism.transitions]


def mechanism_rates(mechanism: BioMechanism) -> list[float]:
    """Rate constants of a mechanism, in transition order."""
    return [float(t.rate_constant) for t in mechanism.transitions]


def simulate_mechanism_meanfield(
    mechanism: BioMechanism, initial: dict[str, float], t: float
) -> dict[str, float]:
    """Mean-field forward prediction for a BioMechanism at time ``t``."""
    species = [s.name for s in mechanism.states]
    return simulate_meanfield(
        species, mechanism_edges(mechanism), mechanism_rates(mechanism), initial, t
    )


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
def gaussian_log_likelihood(
    predicted: dict[str, float],
    observed: dict[str, float],
    uncertainties: Optional[dict[str, float]] = None,
    default_rel_uncertainty: float = 0.1,
    floor: float = 1e-9,
) -> float:
    """Gaussian log-likelihood of ``observed`` given model ``predicted`` values."""
    uncertainties = uncertainties or {}
    ll = 0.0
    for name, obs in observed.items():
        if name not in predicted:
            continue
        sigma = uncertainties.get(name, max(abs(obs) * default_rel_uncertainty, floor))
        if sigma <= 0:
            sigma = floor
        residual = (obs - predicted[name]) / sigma
        ll += -0.5 * residual**2 - 0.5 * np.log(2.0 * np.pi * sigma**2)
    return float(ll)


# --------------------------------------------------------------------------- #
# Model + dataset abstractions for inference
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """A single experimental observation for inference.

    Attributes:
        initial: initial concentrations of (a subset of) species
        time: observation time
        observed: measured concentrations at ``time``
        uncertainties: per-species measurement sigma (optional)
    """

    initial: dict[str, float]
    time: float
    observed: dict[str, float]
    uncertainties: dict[str, float] = field(default_factory=dict)


@dataclass
class MechanismModel:
    """A candidate topology whose rate constants are the inference parameters."""

    species: list[str]
    edges: list[Edge]

    @classmethod
    def from_mechanism(cls, mechanism: BioMechanism) -> "MechanismModel":
        return cls([s.name for s in mechanism.states], mechanism_edges(mechanism))

    @property
    def n_params(self) -> int:
        return len(self.edges)

    def predict(
        self, rates: Sequence[float], initial: dict[str, float], t: float
    ) -> dict[str, float]:
        return simulate_meanfield(self.species, self.edges, rates, initial, t)

    def log_likelihood(
        self, rates: Sequence[float], data: Sequence[Observation]
    ) -> float:
        total = 0.0
        for obs in data:
            predicted = self.predict(rates, obs.initial, obs.time)
            total += gaussian_log_likelihood(predicted, obs.observed, obs.uncertainties)
        return total


# --------------------------------------------------------------------------- #
# Parameter inference: random-walk Metropolis over log10(rates)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceResult:
    """Posterior samples and summaries over a model's rate constants."""

    model: MechanismModel
    samples: np.ndarray  # (n_kept, n_params) in linear rate space
    log_post: np.ndarray  # (n_kept,)
    acceptance_rate: float

    @property
    def posterior_mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def credible_interval(self, level: float = 0.95) -> np.ndarray:
        lo = (1.0 - level) / 2.0 * 100.0
        hi = (1.0 + level) / 2.0 * 100.0
        return np.percentile(self.samples, [lo, hi], axis=0).T  # (n_params, 2)


def metropolis_infer(
    model: MechanismModel,
    data: Sequence[Observation],
    log_rate_bounds: tuple[float, float] = (-4.0, 2.0),
    n_samples: int = 4000,
    burn_in: int = 1000,
    proposal_sigma: float = 0.1,
    seed: Optional[int] = None,
    target_acceptance: float = 0.3,
) -> InferenceResult:
    """Infer rate constants with random-walk Metropolis in log10 space.

    The prior is uniform over ``log_rate_bounds`` for each rate (log-uniform in
    linear space), which is the standard weakly-informative prior for kinetic
    constants spanning orders of magnitude. During burn-in the proposal scale is
    adapted toward ``target_acceptance`` (Robbins-Monro style) so the sampler
    mixes well regardless of how sharply the data constrain the parameters.
    """
    rng = np.random.default_rng(seed)
    n = model.n_params
    lo, hi = log_rate_bounds

    def log_prior(theta: np.ndarray) -> float:
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return 0.0

    def log_posterior(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + model.log_likelihood(10.0**theta, data)

    # Initialize at the prior midpoint.
    theta = np.full(n, (lo + hi) / 2.0)
    current_lp = log_posterior(theta)
    kept_theta: list[np.ndarray] = []
    kept_lp: list[float] = []
    sigma = float(proposal_sigma)
    accepted_kept = 0
    total = n_samples + burn_in

    for step in range(total):
        proposal = theta + rng.normal(0.0, sigma, size=n)
        prop_lp = log_posterior(proposal)
        accept = np.log(rng.random()) < (prop_lp - current_lp)
        if accept:
            theta, current_lp = proposal, prop_lp
        if step < burn_in:
            # Adapt the proposal scale toward the target acceptance rate.
            adjust = 1.0 + (1.0 if accept else -1.0) * (
                (1.0 - target_acceptance) if accept else target_acceptance
            ) / max(1.0, (step + 1) ** 0.5)
            sigma = float(np.clip(sigma * adjust, 1e-3, 2.0))
        else:
            kept_theta.append(theta.copy())
            kept_lp.append(current_lp)
            if accept:
                accepted_kept += 1

    samples = 10.0 ** np.array(kept_theta)
    return InferenceResult(
        model=model,
        samples=samples,
        log_post=np.array(kept_lp),
        acceptance_rate=accepted_kept / max(1, n_samples),
    )


# --------------------------------------------------------------------------- #
# Bayesian model (topology) selection via Monte-Carlo evidence
# --------------------------------------------------------------------------- #
def log_marginal_likelihood(
    model: MechanismModel,
    data: Sequence[Observation],
    log_rate_bounds: tuple[float, float] = (-4.0, 2.0),
    n_samples: int = 20000,
    seed: Optional[int] = None,
) -> float:
    """Monte-Carlo estimate of log P(data | topology), integrating rates over the
    (log-uniform) prior. Uses the log-sum-exp of prior-sampled log-likelihoods.
    """
    rng = np.random.default_rng(seed)
    lo, hi = log_rate_bounds
    n = model.n_params
    if n == 0:
        return model.log_likelihood([], data)
    thetas = rng.uniform(lo, hi, size=(n_samples, n))
    lls = np.array([model.log_likelihood(10.0**theta, data) for theta in thetas])
    m = np.max(lls)
    return float(m + np.log(np.mean(np.exp(lls - m))))


def model_selection(
    models: Sequence[MechanismModel],
    data: Sequence[Observation],
    log_rate_bounds: tuple[float, float] = (-4.0, 2.0),
    n_samples: int = 20000,
    seed: Optional[int] = None,
) -> list[tuple[MechanismModel, float, float]]:
    """Rank candidate topologies by posterior probability (equal model priors).

    Returns a list of (model, log_evidence, posterior_probability) sorted by
    descending posterior probability.
    """
    log_evidences = [
        log_marginal_likelihood(m, data, log_rate_bounds, n_samples, seed)
        for m in models
    ]
    arr = np.array(log_evidences)
    m = np.max(arr)
    weights = np.exp(arr - m)
    posteriors = weights / np.sum(weights)
    ranked = sorted(
        zip(models, log_evidences, posteriors), key=lambda x: x[2], reverse=True
    )
    return ranked


__all__ = [
    "Observation",
    "MechanismModel",
    "InferenceResult",
    "simulate_meanfield",
    "simulate_mechanism_meanfield",
    "gaussian_log_likelihood",
    "metropolis_infer",
    "log_marginal_likelihood",
    "model_selection",
    "mechanism_edges",
    "mechanism_rates",
]

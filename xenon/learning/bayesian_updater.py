"""Bayesian updating engine for mechanism posteriors.

Implements sequential Bayesian inference:
P(mechanism | experiment) ∝ P(experiment | mechanism) × P(mechanism)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.mechanism import BioMechanism


class ExperimentResult:
    """Result from a biological experiment.

    Attributes:
        experiment_type: Type of experiment (e.g., 'concentration', 'kinetics', 'perturbation')
        observations: Observed data points
        uncertainties: Measurement uncertainties
        conditions: Experimental conditions (temperature, pH, etc.)
        metadata: Additional metadata
    """

    def __init__(
        self,
        experiment_type: str,
        observations: dict[str, float],
        uncertainties: Optional[dict[str, float]] = None,
        conditions: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.experiment_type = experiment_type
        self.observations = observations
        self.uncertainties = uncertainties or {}
        self.conditions = conditions or {}
        self.metadata = metadata or {}


class BayesianUpdater:
    """Bayesian updating engine for mechanism learning.

    Core algorithm: P(mechanism | experiment) ∝ P(experiment | mechanism) × P(mechanism)

    Attributes:
        likelihood_scale: Scaling factor for likelihood computation
        evidence_threshold: Minimum posterior probability to keep mechanisms
    """

    def __init__(
        self,
        likelihood_scale: float = 1.0,
        evidence_threshold: float = 1e-6,
    ):
        """Initialize Bayesian updater.

        Args:
            likelihood_scale: Scaling factor for likelihood computation
            evidence_threshold: Minimum posterior to retain mechanisms
        """
        self.likelihood_scale = likelihood_scale
        self.evidence_threshold = evidence_threshold
        self._total_experiments = 0

    def update_mechanisms(
        self,
        mechanisms: list[BioMechanism],
        experiment_result: ExperimentResult,
    ) -> list[BioMechanism]:
        """Update mechanism posteriors based on experimental result.

        Implements Bayesian update:
        P(M|E) = P(E|M) × P(M) / P(E)

        where P(E) = sum over all M of P(E|M) × P(M) (normalization)

        Args:
            mechanisms: List of candidate mechanisms
            experiment_result: Experimental observation

        Returns:
            Updated mechanisms with new posteriors
        """
        if not mechanisms:
            return []

        # Compute likelihoods for each mechanism
        likelihoods = []
        for mechanism in mechanisms:
            likelihood = self.compute_likelihood(mechanism, experiment_result)
            likelihoods.append(likelihood)

        # Bayesian update: posterior ∝ likelihood × prior
        unnormalized_posteriors = []
        for mechanism, likelihood in zip(mechanisms, likelihoods):
            prior = mechanism.posterior  # Current posterior is prior for next update
            unnormalized_posterior = likelihood * prior
            unnormalized_posteriors.append(unnormalized_posterior)

        # Normalize posteriors
        total_evidence = sum(unnormalized_posteriors)

        if total_evidence > 0:
            for mechanism, unnormalized in zip(mechanisms, unnormalized_posteriors):
                mechanism.posterior = unnormalized / total_evidence
                mechanism.provenance.append(
                    f"bayesian_update_{self._total_experiments}:{experiment_result.experiment_type}"
                )

        self._total_experiments += 1

        return mechanisms

    def compute_likelihood(
        self,
        mechanism: BioMechanism,
        experiment: ExperimentResult,
    ) -> float:
        """Compute P(experiment | mechanism).

        Likelihood computation depends on experiment type:
        - concentration: Gaussian likelihood based on predicted vs observed
        - kinetics: Rate constant agreement
        - perturbation: Response prediction accuracy

        Args:
            mechanism: Candidate mechanism
            experiment: Experimental result

        Returns:
            Likelihood value (0 to 1)
        """
        if experiment.experiment_type == "concentration":
            return self._likelihood_concentration(mechanism, experiment)
        elif experiment.experiment_type == "kinetics":
            return self._likelihood_kinetics(mechanism, experiment)
        elif experiment.experiment_type == "perturbation":
            return self._likelihood_perturbation(mechanism, experiment)
        else:
            # Default: uniform likelihood for unknown experiment types
            return 1.0

    def _likelihood_concentration(
        self,
        mechanism: BioMechanism,
        experiment: ExperimentResult,
    ) -> float:
        """Likelihood for concentration measurements.

        Uses Gaussian likelihood: exp(-chi^2 / 2)
        where chi^2 = sum((observed - predicted)^2 / uncertainty^2)
        """
        chi_squared = 0.0
        n_measurements = 0

        for state_name, observed_conc in experiment.observations.items():
            if state_name in mechanism._states:
                predicted_conc = mechanism._states[state_name].concentration

                if predicted_conc is not None:
                    uncertainty = experiment.uncertainties.get(state_name, 0.1 * observed_conc)
                    if uncertainty > 0:
                        residual = (observed_conc - predicted_conc) / uncertainty
                        chi_squared += residual**2
                        n_measurements += 1

        if n_measurements == 0:
            return 0.1  # Low likelihood if no predictions available

        # Gaussian likelihood
        likelihood = np.exp(-chi_squared / (2.0 * n_measurements * self.likelihood_scale))
        return max(likelihood, 1e-10)  # Prevent underflow

    def _likelihood_kinetics(
        self,
        mechanism: BioMechanism,
        experiment: ExperimentResult,
    ) -> float:
        """Likelihood for kinetics measurements.

        Compares observed vs predicted rate constants.
        """
        log_likelihood = 0.0
        n_rates = 0

        for transition in mechanism._transitions:
            transition_key = f"{transition.source}->{transition.target}"

            if transition_key in experiment.observations:
                observed_rate = experiment.observations[transition_key]
                predicted_rate = transition.rate_constant

                # Log-normal likelihood (rate constants span many orders of magnitude)
                if observed_rate > 0 and predicted_rate > 0:
                    log_ratio = np.log(predicted_rate / observed_rate)
                    # Assume 1 order of magnitude uncertainty
                    log_likelihood += -0.5 * (log_ratio / np.log(10)) ** 2
                    n_rates += 1

        if n_rates == 0:
            return 0.1

        likelihood = np.exp(log_likelihood / n_rates)
        return max(likelihood, 1e-10)

    def _likelihood_perturbation(
        self,
        mechanism: BioMechanism,
        experiment: ExperimentResult,
    ) -> float:
        """Likelihood for perturbation experiments.

        Evaluates whether mechanism topology supports observed response.
        """
        # Check if perturbation source and target are in mechanism
        source = experiment.conditions.get("perturbation_source")
        target = experiment.conditions.get("perturbation_target")

        if source and target and source in mechanism._states and target in mechanism._states:
            # Check if causal path exists
            paths = mechanism.get_causal_paths(source, target)

            if len(paths) > 0:
                # Likelihood increases with number of paths
                likelihood = 1.0 - np.exp(-len(paths) / 5.0)
                return likelihood

        return 0.1  # Low likelihood if topology doesn't support perturbation

    def prune_low_evidence(
        self,
        mechanisms: list[BioMechanism],
        threshold: float | None = None,
    ) -> list[BioMechanism]:
        """Remove mechanisms with posterior below threshold.

        Args:
            mechanisms: List of mechanisms
            threshold: Minimum posterior to retain (uses self.evidence_threshold if None)

        Returns:
            Filtered list of mechanisms
        """
        if threshold is None:
            threshold = self.evidence_threshold

        pruned = [m for m in mechanisms if m.posterior >= threshold]

        return pruned

    def get_evidence_summary(self, mechanisms: list[BioMechanism]) -> dict[str, Any]:
        """Compute summary statistics of mechanism evidence.

        Args:
            mechanisms: List of mechanisms

        Returns:
            Dictionary with summary statistics
        """
        if not mechanisms:
            return {
                "n_mechanisms": 0,
                "total_evidence": 0.0,
                "max_posterior": 0.0,
                "mean_posterior": 0.0,
                "entropy": 0.0,
            }

        posteriors = [m.posterior for m in mechanisms]
        total_evidence = sum(posteriors)

        # Compute entropy (uncertainty in mechanism distribution)
        entropy = 0.0
        for p in posteriors:
            if p > 0:
                entropy += -p * np.log(p)

        return {
            "n_mechanisms": len(mechanisms),
            "total_evidence": total_evidence,
            "max_posterior": max(posteriors),
            "mean_posterior": np.mean(posteriors),
            "entropy": entropy,
        }

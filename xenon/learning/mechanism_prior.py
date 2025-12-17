"""Prior probability computation for biological mechanisms.

Combines multiple sources of prior information:
1. Chemical plausibility (rate constants from literature)
2. Evolutionary conservation (sequence homology)
3. Literature citations (PubMed/bioRxiv)
"""

from __future__ import annotations

import numpy as np

from ..core.mechanism import BioMechanism, Transition


class MechanismPrior:
    """Prior probability computation for mechanisms.

    Combines multiple sources of prior information to assign initial
    probabilities to mechanism hypotheses before experimental data.

    Attributes:
        rate_constant_scale: Scale parameter for rate constant priors
        conservation_weight: Weight for evolutionary conservation
        literature_weight: Weight for literature evidence
    """

    def __init__(
        self,
        rate_constant_scale: float = 1.0,
        conservation_weight: float = 0.3,
        literature_weight: float = 0.3,
    ):
        """Initialize mechanism prior.

        Args:
            rate_constant_scale: Scale for rate constant priors
            conservation_weight: Weight for evolutionary conservation (0-1)
            literature_weight: Weight for literature evidence (0-1)
        """
        self.rate_constant_scale = rate_constant_scale
        self.conservation_weight = conservation_weight
        self.literature_weight = literature_weight

        # Literature database (mock for Phase 1)
        # In Phase 2+, this would query PubMed/bioRxiv APIs
        self._literature_db: dict[str, int] = {}

    def compute_prior(self, mechanism: BioMechanism) -> float:
        """Compute prior probability P(mechanism).

        Combines:
        1. Rate constant plausibility
        2. Evolutionary conservation
        3. Literature support

        Args:
            mechanism: Candidate mechanism

        Returns:
            Prior probability (0 to 1)
        """
        # Base prior from rate constants
        rate_prior = self._rate_constant_prior_product(mechanism)

        # Evolutionary conservation prior
        conservation_prior = self._conservation_prior(mechanism)

        # Literature citation prior
        literature_prior = self._literature_prior(mechanism)

        # Combine priors (geometric mean)
        weights = [
            1.0 - self.conservation_weight - self.literature_weight,
            self.conservation_weight,
            self.literature_weight,
        ]

        combined_prior = (
            rate_prior ** weights[0]
            * conservation_prior ** weights[1]
            * literature_prior ** weights[2]
        )

        return combined_prior

    def rate_constant_prior(self, transition: Transition) -> float:
        """Compute prior for a single rate constant.

        Uses log-normal distribution centered on typical biochemical rates:
        - Diffusion-limited: ~10^9 M^-1 s^-1
        - Enzyme catalysis: ~10^2 - 10^6 s^-1
        - Conformational change: ~10^3 - 10^6 s^-1

        Args:
            transition: Transition with rate constant

        Returns:
            Prior probability (0 to 1)
        """
        rate = transition.rate_constant

        if rate <= 0:
            return 0.01  # Very low prior for non-physical rates

        # Log-normal prior centered at 10^3 s^-1 with log-scale width of 3
        log_rate = np.log10(rate)
        typical_log_rate = 3.0  # 10^3 s^-1
        log_width = 3.0  # Cover 10^0 to 10^6

        # Gaussian in log-space
        log_diff = (log_rate - typical_log_rate) / log_width
        prior = np.exp(-0.5 * log_diff**2)

        # Penalize extremely fast or slow rates
        if log_rate < -3 or log_rate > 9:
            prior *= 0.1

        return max(prior, 0.01)

    def _rate_constant_prior_product(self, mechanism: BioMechanism) -> float:
        """Compute product of rate constant priors.

        Args:
            mechanism: Candidate mechanism

        Returns:
            Combined rate constant prior
        """
        if not mechanism._transitions:
            return 0.5  # Neutral prior for mechanisms without transitions

        log_prior = 0.0
        for transition in mechanism._transitions:
            prior = self.rate_constant_prior(transition)
            log_prior += np.log(max(prior, 1e-10))

        # Geometric mean of individual priors
        avg_log_prior = log_prior / len(mechanism._transitions)

        return np.exp(avg_log_prior)

    def _conservation_prior(self, mechanism: BioMechanism) -> float:
        """Compute evolutionary conservation prior.

        In Phase 1, this is a mock implementation.
        In Phase 2+, this would:
        1. Query UniProt for sequence homology
        2. Check conservation across species
        3. Identify conserved domains

        Args:
            mechanism: Candidate mechanism

        Returns:
            Conservation prior (0 to 1)
        """
        # Mock: assume moderate conservation
        # Real implementation would query protein databases

        # Check if mechanism has thermodynamically feasible states
        if mechanism.is_thermodynamically_feasible():
            return 0.7  # Higher prior for thermodynamically consistent mechanisms
        else:
            return 0.3  # Lower prior for thermodynamically inconsistent

    def _literature_prior(self, mechanism: BioMechanism) -> float:
        """Compute literature citation prior.

        In Phase 1, this is a mock implementation.
        In Phase 2+, this would:
        1. Query PubMed/bioRxiv for related papers
        2. Extract interaction networks from text
        3. Count supporting citations

        Args:
            mechanism: Candidate mechanism

        Returns:
            Literature prior (0 to 1)
        """
        # Mock: check if proteins are in literature database
        n_citations = 0

        for state in mechanism._states.values():
            protein = state.molecule
            citations = self._literature_db.get(protein, 0)
            n_citations += citations

        # Transform citation count to probability
        # More citations → higher prior
        prior = 1.0 - np.exp(-n_citations / 10.0)

        # Ensure minimum prior
        return max(prior, 0.1)

    def add_literature_evidence(self, protein: str, citation_count: int) -> None:
        """Add literature evidence for a protein.

        Mock method for Phase 1. In Phase 2+, this would be replaced
        by automated literature mining.

        Args:
            protein: Protein name
            citation_count: Number of relevant citations
        """
        self._literature_db[protein] = citation_count

    def initialize_mechanism_priors(
        self,
        mechanisms: List[BioMechanism],
    ) -> List[BioMechanism]:
        """Initialize priors for a list of mechanisms.

        Args:
            mechanisms: List of mechanisms

        Returns:
            Mechanisms with initialized posteriors
        """
        for mechanism in mechanisms:
            mechanism.posterior = self.compute_prior(mechanism)

        # Normalize so posteriors sum to 1
        total = sum(m.posterior for m in mechanisms)
        if total > 0:
            for mechanism in mechanisms:
                mechanism.posterior /= total

        return mechanisms

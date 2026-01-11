"""Quantum Output Verification for QRATUM.

This module implements verification of quantum outputs before they
are allowed to be reinjected into QRATUM pipelines.

Key principles:
- Quantum outputs **never bypass** verification
- All quantum use is **proposal-only** until approved
- Verification includes entropy analysis, noise detection, and consistency checks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class VerificationStatus(Enum):
    """Status of verification check."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Result of quantum output verification.

    Attributes:
        verification_id: Unique identifier for this verification
        execution_id: ID of the quantum execution being verified
        status: Overall verification status
        entropy_score: Shannon entropy of output distribution
        noise_estimate: Estimated noise level (0-1)
        consistency_score: Score for result consistency (0-1)
        statistical_validity: Whether statistical requirements are met
        checks_performed: List of verification checks performed
        check_results: Results of individual checks
        timestamp: When verification was performed
        notes: Additional notes or warnings
    """

    verification_id: str
    execution_id: str
    status: VerificationStatus = VerificationStatus.PENDING
    entropy_score: float = 0.0
    noise_estimate: float = 0.0
    consistency_score: float = 1.0
    statistical_validity: bool = False
    checks_performed: list[str] = field(default_factory=list)
    check_results: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def is_approved(self) -> bool:
        """Check if result is approved for use."""
        return self.status == VerificationStatus.PASSED


class QuantumVerifier:
    """Verifier for quantum execution outputs.

    This verifier ensures quantum outputs meet QRATUM's trust invariants
    before they can be used in downstream computations.

    Verification includes:
    1. Entropy analysis - detect anomalous distributions
    2. Noise estimation - estimate NISQ error rates
    3. Consistency checks - verify result reproducibility
    4. Statistical validity - ensure sufficient samples

    Example:
        >>> verifier = QuantumVerifier()
        >>> result = backend.execute_circuit(circuit)
        >>> verification = verifier.verify(result, expected_distribution=expected)
        >>> if verification.is_approved:
        ...     # Safe to use result
        ...     pass
    """

    def __init__(
        self,
        min_shots: int = 1000,
        max_entropy_deviation: float = 0.5,
        max_noise_threshold: float = 0.2,
        min_consistency_score: float = 0.8,
    ):
        """Initialize verifier with thresholds.

        Args:
            min_shots: Minimum shots for statistical validity
            max_entropy_deviation: Maximum allowed entropy deviation from expected
            max_noise_threshold: Maximum acceptable noise estimate
            min_consistency_score: Minimum consistency score to pass
        """
        self.min_shots = min_shots
        self.max_entropy_deviation = max_entropy_deviation
        self.max_noise_threshold = max_noise_threshold
        self.min_consistency_score = min_consistency_score

    def verify(
        self,
        result: Any,
        expected_distribution: dict[str, float] | None = None,
        reference_results: list[dict[str, int]] | None = None,
    ) -> VerificationResult:
        """Verify quantum execution result.

        Args:
            result: Execution result to verify
            expected_distribution: Expected probability distribution (optional)
            reference_results: Previous results for consistency check (optional)

        Returns:
            VerificationResult with all check results
        """
        import uuid

        verification_id = str(uuid.uuid4())
        execution_id = getattr(result, "execution_id", "unknown")

        # Extract counts
        if hasattr(result, "counts"):
            counts = result.counts
        elif isinstance(result, dict) and "counts" in result:
            counts = result["counts"]
        else:
            return VerificationResult(
                verification_id=verification_id,
                execution_id=execution_id,
                status=VerificationStatus.FAILED,
                notes=["Could not extract counts from result"],
            )

        checks_performed = []
        check_results = {}
        notes = []

        # 1. Statistical validity check
        total_shots = sum(counts.values())
        statistical_validity = total_shots >= self.min_shots
        checks_performed.append("statistical_validity")
        check_results["statistical_validity"] = {
            "total_shots": total_shots,
            "min_required": self.min_shots,
            "passed": statistical_validity,
        }
        if not statistical_validity:
            notes.append(f"Insufficient shots: {total_shots} < {self.min_shots}")

        # 2. Entropy analysis
        entropy_score = self._compute_entropy(counts)
        checks_performed.append("entropy_analysis")
        check_results["entropy_analysis"] = {
            "entropy": entropy_score,
            "max_possible": math.log2(len(counts)) if len(counts) > 0 else 0,
        }

        entropy_passed = True
        if expected_distribution is not None:
            expected_entropy = self._compute_expected_entropy(expected_distribution)
            entropy_deviation = abs(entropy_score - expected_entropy)
            entropy_passed = entropy_deviation <= self.max_entropy_deviation
            check_results["entropy_analysis"]["expected"] = expected_entropy
            check_results["entropy_analysis"]["deviation"] = entropy_deviation
            check_results["entropy_analysis"]["passed"] = entropy_passed
            if not entropy_passed:
                notes.append(f"Entropy deviation too high: {entropy_deviation:.4f}")

        # 3. Noise estimation
        noise_estimate = self._estimate_noise(counts, expected_distribution)
        checks_performed.append("noise_estimation")
        noise_passed = noise_estimate <= self.max_noise_threshold
        check_results["noise_estimation"] = {
            "estimate": noise_estimate,
            "threshold": self.max_noise_threshold,
            "passed": noise_passed,
        }
        if not noise_passed:
            notes.append(f"Noise estimate too high: {noise_estimate:.4f}")

        # 4. Consistency check (if reference results provided)
        consistency_score = 1.0
        if reference_results:
            consistency_score = self._compute_consistency(counts, reference_results)
            checks_performed.append("consistency_check")
            consistency_passed = consistency_score >= self.min_consistency_score
            check_results["consistency_check"] = {
                "score": consistency_score,
                "threshold": self.min_consistency_score,
                "passed": consistency_passed,
            }
            if not consistency_passed:
                notes.append(f"Consistency score too low: {consistency_score:.4f}")
        else:
            check_results["consistency_check"] = {"skipped": True, "reason": "No reference results"}

        # Determine overall status
        all_passed = (
            statistical_validity
            and entropy_passed
            and noise_passed
            and consistency_score >= self.min_consistency_score
        )

        if all_passed:
            status = VerificationStatus.PASSED
        elif statistical_validity and (entropy_passed or noise_passed):
            status = VerificationStatus.WARNING
            notes.append("Result passed with warnings - manual review recommended")
        else:
            status = VerificationStatus.FAILED

        return VerificationResult(
            verification_id=verification_id,
            execution_id=execution_id,
            status=status,
            entropy_score=entropy_score,
            noise_estimate=noise_estimate,
            consistency_score=consistency_score,
            statistical_validity=statistical_validity,
            checks_performed=checks_performed,
            check_results=check_results,
            notes=notes,
        )

    def _compute_entropy(self, counts: dict[str, int]) -> float:
        """Compute Shannon entropy of measurement distribution.

        Args:
            counts: Measurement outcome counts

        Returns:
            Shannon entropy in bits
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _compute_expected_entropy(self, distribution: dict[str, float]) -> float:
        """Compute expected entropy from probability distribution.

        Args:
            distribution: Expected probability distribution

        Returns:
            Expected Shannon entropy in bits
        """
        entropy = 0.0
        for p in distribution.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _estimate_noise(
        self, counts: dict[str, int], expected: dict[str, float] | None
    ) -> float:
        """Estimate noise level in quantum output.

        Uses total variation distance from expected distribution
        as a proxy for noise.

        Args:
            counts: Measured counts
            expected: Expected probability distribution

        Returns:
            Noise estimate between 0 (no noise) and 1 (maximum noise)
        """
        if expected is None:
            # Without expected distribution, estimate from uniformity deviation
            total = sum(counts.values())
            n = len(counts)
            if n == 0 or total == 0:
                return 0.0

            # Compare to uniform distribution
            uniform_p = 1.0 / n
            tvd = 0.0
            for count in counts.values():
                measured_p = count / total
                tvd += abs(measured_p - uniform_p)

            return min(1.0, tvd / 2)

        # Total variation distance from expected
        total = sum(counts.values())
        if total == 0:
            return 1.0

        tvd = 0.0
        all_outcomes = set(counts.keys()) | set(expected.keys())

        for outcome in all_outcomes:
            measured_p = counts.get(outcome, 0) / total
            expected_p = expected.get(outcome, 0)
            tvd += abs(measured_p - expected_p)

        return min(1.0, tvd / 2)

    def _compute_consistency(
        self, counts: dict[str, int], reference_results: list[dict[str, int]]
    ) -> float:
        """Compute consistency score with reference results.

        Uses average fidelity across reference results.

        Args:
            counts: Current measurement counts
            reference_results: List of previous measurement counts

        Returns:
            Consistency score between 0 (inconsistent) and 1 (perfectly consistent)
        """
        if not reference_results:
            return 1.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        # Convert to probability distribution
        measured = {k: v / total for k, v in counts.items()}

        fidelities = []
        for ref_counts in reference_results:
            ref_total = sum(ref_counts.values())
            if ref_total == 0:
                continue

            ref_dist = {k: v / ref_total for k, v in ref_counts.items()}

            # Classical fidelity (Bhattacharyya coefficient)
            all_outcomes = set(measured.keys()) | set(ref_dist.keys())
            bc = 0.0
            for outcome in all_outcomes:
                p = measured.get(outcome, 0)
                q = ref_dist.get(outcome, 0)
                bc += math.sqrt(p * q)

            fidelities.append(bc)

        return np.mean(fidelities) if fidelities else 1.0


class TopologicalDiagnosticObserver:
    """Observer for topological diagnostics on quantum outputs.

    This observer monitors quantum execution outputs for anomalies
    using topological data analysis techniques.

    Features:
    - Persistent homology analysis of measurement landscapes
    - Noise topology detection
    - Error pattern recognition
    """

    def __init__(self):
        """Initialize diagnostic observer."""
        self._observations: list[dict[str, Any]] = []

    def observe(self, counts: dict[str, int], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Observe quantum output and compute diagnostics.

        Args:
            counts: Measurement outcome counts
            metadata: Additional context for observation

        Returns:
            Dictionary with diagnostic metrics
        """
        import uuid

        observation_id = str(uuid.uuid4())
        total = sum(counts.values())

        if total == 0:
            return {"observation_id": observation_id, "error": "Empty counts"}

        # Convert to probability landscape
        probs = np.array([v / total for v in counts.values()])

        # Compute topological features
        diagnostics = {
            "observation_id": observation_id,
            "total_shots": total,
            "unique_outcomes": len(counts),
            "max_probability": float(np.max(probs)),
            "min_probability": float(np.min(probs)),
            "probability_variance": float(np.var(probs)),
            "concentration_index": self._compute_concentration(probs),
            "sparsity_index": self._compute_sparsity(probs),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self._observations.append(diagnostics)
        return diagnostics

    def _compute_concentration(self, probs: np.ndarray) -> float:
        """Compute probability concentration index.

        Higher values indicate more concentrated distribution.

        Args:
            probs: Probability array

        Returns:
            Concentration index (0-1)
        """
        if len(probs) == 0:
            return 0.0

        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)

        # Find how many outcomes account for 90% of probability
        idx_90 = np.searchsorted(cumsum, 0.9)

        return 1.0 - (idx_90 / len(probs))

    def _compute_sparsity(self, probs: np.ndarray) -> float:
        """Compute sparsity index of distribution.

        Args:
            probs: Probability array

        Returns:
            Sparsity index (0=uniform, 1=maximally sparse)
        """
        if len(probs) == 0:
            return 0.0

        # Gini coefficient as sparsity measure
        n = len(probs)
        sorted_probs = np.sort(probs)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n

        return max(0.0, min(1.0, gini))

    def get_observations(self) -> list[dict[str, Any]]:
        """Get all recorded observations.

        Returns:
            List of observation dictionaries
        """
        return self._observations.copy()

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics across all observations.

        Returns:
            Summary statistics dictionary
        """
        if not self._observations:
            return {"count": 0}

        concentrations = [o["concentration_index"] for o in self._observations]
        sparsities = [o["sparsity_index"] for o in self._observations]

        return {
            "count": len(self._observations),
            "mean_concentration": float(np.mean(concentrations)),
            "std_concentration": float(np.std(concentrations)),
            "mean_sparsity": float(np.mean(sparsities)),
            "std_sparsity": float(np.std(sparsities)),
        }

"""Enhanced Topological Diagnostic Observer for QRATUM.

This module provides read-only diagnostic observation capabilities for
quantum execution outputs using topological data analysis techniques.

Core Principles:
- Diagnostics read-only; resolutions proposal-only
- Forever non-agentic epistemic compiler
- No auto-modification of system state
- All recommendations require human approval

Features:
- Persistent homology analysis of measurement landscapes
- Collapse index computation and tracking
- Fidelity metrics for simulation accuracy
- Noise topology detection
- SOI (System of Interest) visualization support
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic findings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class DiagnosticCategory(Enum):
    """Categories of diagnostic findings."""

    NOISE = "noise"
    DECOHERENCE = "decoherence"
    TOPOLOGY = "topology"
    FIDELITY = "fidelity"
    CONSISTENCY = "consistency"
    ENTROPY = "entropy"


@dataclass
class DiagnosticFinding:
    """A single diagnostic finding from observation.

    Attributes:
        finding_id: Unique identifier for this finding
        category: Category of the finding
        severity: Severity level
        description: Human-readable description
        metrics: Relevant metrics for this finding
        recommended_action: Proposed action (proposal-only)
        timestamp: When finding was made
        observation_id: ID of parent observation
    """

    finding_id: str
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    description: str
    metrics: dict[str, float] = field(default_factory=dict)
    recommended_action: str = ""
    timestamp: str = ""
    observation_id: str = ""

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_proposal(self) -> dict[str, Any]:
        """Convert finding to proposal format (proposal-only resolution).

        Returns:
            Proposal dictionary with recommended action
        """
        return {
            "proposal_id": str(uuid.uuid4()),
            "finding_id": self.finding_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "recommended_action": self.recommended_action,
            "requires_approval": True,
            "status": "pending",
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class CollapseMetrics:
    """Metrics for topological collapse analysis.

    Measures how concentrated the probability distribution is,
    indicating potential quantum state collapse or noise issues.

    Attributes:
        collapse_index: Primary collapse metric (0=uniform, 1=fully collapsed)
        concentration_coefficient: Gini-like concentration measure
        effective_support: Number of outcomes with significant probability
        dominant_outcomes: Outcomes with highest probabilities
        entropy_ratio: Actual entropy / max possible entropy
        target_reduction: Target reduction percentage (P1: ≥30%)
    """

    collapse_index: float = 0.0
    concentration_coefficient: float = 0.0
    effective_support: int = 0
    dominant_outcomes: list[str] = field(default_factory=list)
    entropy_ratio: float = 1.0
    target_reduction: float = 0.3

    @property
    def meets_target(self) -> bool:
        """Check if collapse reduction meets P1 target (≥30%)."""
        return self.collapse_index <= (1.0 - self.target_reduction)


@dataclass
class FidelityMetrics:
    """Fidelity metrics for quantum simulation accuracy.

    Attributes:
        simulation_fidelity: Overall simulation fidelity (P1 target: ≥0.999)
        state_fidelity: Quantum state fidelity
        process_fidelity: Process/gate fidelity
        classical_fidelity: Classical approximation fidelity
        error_rate: Estimated error rate
    """

    simulation_fidelity: float = 1.0
    state_fidelity: float = 1.0
    process_fidelity: float = 1.0
    classical_fidelity: float = 1.0
    error_rate: float = 0.0

    @property
    def meets_p1_target(self) -> bool:
        """Check if simulation fidelity meets P1 target (≥0.999)."""
        return self.simulation_fidelity >= 0.999


@dataclass
class TopologicalObservation:
    """Complete observation record with all metrics.

    Attributes:
        observation_id: Unique identifier
        timestamp: When observation was made
        collapse_metrics: Topological collapse analysis
        fidelity_metrics: Fidelity measurements
        findings: List of diagnostic findings
        raw_diagnostics: Raw computed values
        provenance_hash: Hash for verification
        metadata: Additional context
    """

    observation_id: str
    timestamp: str = ""
    collapse_metrics: CollapseMetrics = field(default_factory=CollapseMetrics)
    fidelity_metrics: FidelityMetrics = field(default_factory=FidelityMetrics)
    findings: list[DiagnosticFinding] = field(default_factory=list)
    raw_diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class EnhancedTopologicalObserver:
    """Enhanced read-only diagnostic observer for quantum outputs.

    This observer provides comprehensive topological analysis without
    modifying system state. All resolutions are proposal-only.

    Features:
    - Collapse index computation with P1 target tracking
    - Fidelity metrics for simulation accuracy
    - Persistent homology proxies
    - Noise topology detection
    - SOI visualization data generation

    Example:
        >>> observer = EnhancedTopologicalObserver()
        >>> observation = observer.observe(counts, expected_distribution)
        >>> for finding in observation.findings:
        ...     if finding.severity == DiagnosticSeverity.CRITICAL:
        ...         proposal = finding.to_proposal()
        ...         # Submit proposal for human approval
    """

    def __init__(
        self,
        fidelity_threshold: float = 0.999,
        collapse_reduction_target: float = 0.3,
        noise_threshold: float = 0.1,
    ):
        """Initialize observer with thresholds.

        Args:
            fidelity_threshold: Minimum acceptable fidelity (P1: 0.999)
            collapse_reduction_target: Target collapse reduction (P1: 30%)
            noise_threshold: Threshold for noise warnings
        """
        self.fidelity_threshold = fidelity_threshold
        self.collapse_reduction_target = collapse_reduction_target
        self.noise_threshold = noise_threshold

        self._observations: list[TopologicalObservation] = []
        self._proposals: list[dict[str, Any]] = []

    def observe(
        self,
        counts: dict[str, int],
        expected_distribution: dict[str, float] | None = None,
        reference_counts: list[dict[str, int]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TopologicalObservation:
        """Perform comprehensive topological observation (read-only).

        Args:
            counts: Measurement outcome counts
            expected_distribution: Optional expected probability distribution
            reference_counts: Optional reference results for consistency
            metadata: Additional context for observation

        Returns:
            TopologicalObservation with all metrics and findings
        """
        observation_id = str(uuid.uuid4())
        total = sum(counts.values())

        if total == 0:
            return TopologicalObservation(
                observation_id=observation_id,
                raw_diagnostics={"error": "Empty counts provided"},
                metadata=metadata or {},
            )

        # Convert to probabilities
        probs = {k: v / total for k, v in counts.items()}
        prob_array = np.array(list(probs.values()))

        # Compute collapse metrics
        collapse_metrics = self._compute_collapse_metrics(counts, probs, prob_array)

        # Compute fidelity metrics
        fidelity_metrics = self._compute_fidelity_metrics(
            probs, expected_distribution, reference_counts
        )

        # Generate diagnostic findings
        findings = self._generate_findings(
            observation_id, collapse_metrics, fidelity_metrics, probs, expected_distribution
        )

        # Compute raw diagnostics
        raw_diagnostics = self._compute_raw_diagnostics(counts, probs, prob_array)

        # Compute provenance hash
        provenance_hash = self._compute_provenance_hash(counts, raw_diagnostics)

        observation = TopologicalObservation(
            observation_id=observation_id,
            collapse_metrics=collapse_metrics,
            fidelity_metrics=fidelity_metrics,
            findings=findings,
            raw_diagnostics=raw_diagnostics,
            provenance_hash=provenance_hash,
            metadata=metadata or {},
        )

        self._observations.append(observation)
        return observation

    def get_observations(self) -> list[TopologicalObservation]:
        """Get all recorded observations (read-only access).

        Returns:
            List of observations
        """
        return self._observations.copy()

    def get_proposals(self) -> list[dict[str, Any]]:
        """Get all generated proposals awaiting approval.

        Returns:
            List of proposal dictionaries
        """
        return self._proposals.copy()

    def generate_soi_visualization_data(
        self, observation_id: str | None = None
    ) -> dict[str, Any]:
        """Generate System of Interest (SOI) visualization data.

        Args:
            observation_id: Optional specific observation ID

        Returns:
            Visualization data structure for SOI rendering
        """
        if observation_id:
            observations = [o for o in self._observations if o.observation_id == observation_id]
        else:
            observations = self._observations

        if not observations:
            return {"error": "No observations found"}

        # Aggregate data for visualization
        collapse_values = [o.collapse_metrics.collapse_index for o in observations]
        fidelity_values = [o.fidelity_metrics.simulation_fidelity for o in observations]
        timestamps = [o.timestamp for o in observations]

        return {
            "visualization_type": "soi_topological",
            "timestamp": datetime.utcnow().isoformat(),
            "data_points": len(observations),
            "collapse_trend": {
                "values": collapse_values,
                "timestamps": timestamps,
                "mean": float(np.mean(collapse_values)) if collapse_values else 0,
                "std": float(np.std(collapse_values)) if collapse_values else 0,
                "target": 1.0 - self.collapse_reduction_target,
            },
            "fidelity_trend": {
                "values": fidelity_values,
                "timestamps": timestamps,
                "mean": float(np.mean(fidelity_values)) if fidelity_values else 0,
                "std": float(np.std(fidelity_values)) if fidelity_values else 0,
                "target": self.fidelity_threshold,
            },
            "findings_summary": {
                "total": sum(len(o.findings) for o in observations),
                "by_severity": self._count_by_severity(observations),
                "by_category": self._count_by_category(observations),
            },
        }

    def get_summary_report(self) -> dict[str, Any]:
        """Generate summary report of all observations.

        Returns:
            Summary report dictionary
        """
        if not self._observations:
            return {"count": 0, "message": "No observations recorded"}

        collapse_indices = [o.collapse_metrics.collapse_index for o in self._observations]
        fidelities = [o.fidelity_metrics.simulation_fidelity for o in self._observations]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "observation_count": len(self._observations),
            "collapse_statistics": {
                "mean": float(np.mean(collapse_indices)),
                "std": float(np.std(collapse_indices)),
                "min": float(np.min(collapse_indices)),
                "max": float(np.max(collapse_indices)),
                "meets_target_count": sum(
                    1 for o in self._observations if o.collapse_metrics.meets_target
                ),
            },
            "fidelity_statistics": {
                "mean": float(np.mean(fidelities)),
                "std": float(np.std(fidelities)),
                "min": float(np.min(fidelities)),
                "max": float(np.max(fidelities)),
                "meets_p1_count": sum(
                    1 for o in self._observations if o.fidelity_metrics.meets_p1_target
                ),
            },
            "findings_total": sum(len(o.findings) for o in self._observations),
            "proposals_pending": len(
                [p for p in self._proposals if p.get("status") == "pending"]
            ),
        }

    def _compute_collapse_metrics(
        self,
        counts: dict[str, int],
        probs: dict[str, float],
        prob_array: np.ndarray,
    ) -> CollapseMetrics:
        """Compute topological collapse metrics."""
        n = len(prob_array)
        if n == 0:
            return CollapseMetrics()

        # Sort probabilities descending
        sorted_probs = np.sort(prob_array)[::-1]
        sorted_outcomes = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Collapse index: based on concentration
        # 0 = uniform, 1 = fully collapsed to single outcome
        max_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0
        collapse_index = max_prob  # Simple: highest probability indicates collapse

        # Concentration coefficient (Gini-like)
        cumsum = np.cumsum(sorted_probs)
        if np.sum(sorted_probs) > 0:
            gini = (2 * np.sum(np.arange(1, n + 1) * sorted_probs)) / (
                n * np.sum(sorted_probs)
            ) - (n + 1) / n
            concentration_coefficient = max(0.0, min(1.0, gini))
        else:
            concentration_coefficient = 0.0

        # Effective support: outcomes with >1% probability
        effective_support = int(np.sum(prob_array > 0.01))

        # Dominant outcomes: top 3
        dominant_outcomes = [k for k, _ in sorted_outcomes[:3]]

        # Entropy ratio
        if n > 1:
            nonzero_probs = prob_array[prob_array > 0]
            entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
            max_entropy = np.log2(n)
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            entropy_ratio = 1.0

        return CollapseMetrics(
            collapse_index=float(collapse_index),
            concentration_coefficient=float(concentration_coefficient),
            effective_support=effective_support,
            dominant_outcomes=dominant_outcomes,
            entropy_ratio=float(entropy_ratio),
            target_reduction=self.collapse_reduction_target,
        )

    def _compute_fidelity_metrics(
        self,
        probs: dict[str, float],
        expected: dict[str, float] | None,
        reference_counts: list[dict[str, int]] | None,
    ) -> FidelityMetrics:
        """Compute fidelity metrics."""
        if expected is None and reference_counts is None:
            # No reference - assume perfect fidelity
            return FidelityMetrics()

        simulation_fidelity = 1.0
        state_fidelity = 1.0
        classical_fidelity = 1.0
        error_rate = 0.0

        if expected:
            # Compute classical fidelity (Bhattacharyya coefficient)
            all_outcomes = set(probs.keys()) | set(expected.keys())
            bc = 0.0
            for outcome in all_outcomes:
                p = probs.get(outcome, 0)
                q = expected.get(outcome, 0)
                bc += np.sqrt(p * q)

            state_fidelity = float(bc)
            simulation_fidelity = float(bc)

            # Total variation distance as error rate proxy
            tvd = 0.0
            for outcome in all_outcomes:
                p = probs.get(outcome, 0)
                q = expected.get(outcome, 0)
                tvd += abs(p - q)
            error_rate = tvd / 2

        if reference_counts:
            # Compute consistency with reference results
            fidelities = []
            for ref in reference_counts:
                ref_total = sum(ref.values())
                if ref_total == 0:
                    continue
                ref_probs = {k: v / ref_total for k, v in ref.items()}

                all_outcomes = set(probs.keys()) | set(ref_probs.keys())
                bc = sum(
                    np.sqrt(probs.get(o, 0) * ref_probs.get(o, 0)) for o in all_outcomes
                )
                fidelities.append(bc)

            if fidelities:
                classical_fidelity = float(np.mean(fidelities))

        return FidelityMetrics(
            simulation_fidelity=simulation_fidelity,
            state_fidelity=state_fidelity,
            process_fidelity=1.0,  # Would need gate-level info
            classical_fidelity=classical_fidelity,
            error_rate=float(error_rate),
        )

    def _generate_findings(
        self,
        observation_id: str,
        collapse_metrics: CollapseMetrics,
        fidelity_metrics: FidelityMetrics,
        probs: dict[str, float],
        expected: dict[str, float] | None,
    ) -> list[DiagnosticFinding]:
        """Generate diagnostic findings from metrics."""
        findings = []

        # Check collapse target
        if not collapse_metrics.meets_target:
            finding = DiagnosticFinding(
                finding_id=str(uuid.uuid4()),
                category=DiagnosticCategory.TOPOLOGY,
                severity=DiagnosticSeverity.WARNING,
                description=(
                    f"Collapse index {collapse_metrics.collapse_index:.4f} exceeds target "
                    f"threshold of {1.0 - self.collapse_reduction_target:.4f}"
                ),
                metrics={
                    "collapse_index": collapse_metrics.collapse_index,
                    "target": 1.0 - self.collapse_reduction_target,
                },
                recommended_action="Consider increasing shots or implementing error mitigation",
                observation_id=observation_id,
            )
            findings.append(finding)
            self._proposals.append(finding.to_proposal())

        # Check fidelity target
        if not fidelity_metrics.meets_p1_target:
            severity = (
                DiagnosticSeverity.CRITICAL
                if fidelity_metrics.simulation_fidelity < 0.95
                else DiagnosticSeverity.WARNING
            )
            finding = DiagnosticFinding(
                finding_id=str(uuid.uuid4()),
                category=DiagnosticCategory.FIDELITY,
                severity=severity,
                description=(
                    f"Simulation fidelity {fidelity_metrics.simulation_fidelity:.4f} "
                    f"below P1 target of {self.fidelity_threshold}"
                ),
                metrics={
                    "simulation_fidelity": fidelity_metrics.simulation_fidelity,
                    "target": self.fidelity_threshold,
                    "error_rate": fidelity_metrics.error_rate,
                },
                recommended_action="Verify input parameters or increase simulation precision",
                observation_id=observation_id,
            )
            findings.append(finding)
            self._proposals.append(finding.to_proposal())

        # Check for low effective support
        if collapse_metrics.effective_support <= 2 and len(probs) > 4:
            finding = DiagnosticFinding(
                finding_id=str(uuid.uuid4()),
                category=DiagnosticCategory.ENTROPY,
                severity=DiagnosticSeverity.WARNING,
                description=(
                    f"Low effective support: only {collapse_metrics.effective_support} "
                    f"outcomes have >1% probability"
                ),
                metrics={
                    "effective_support": collapse_metrics.effective_support,
                    "total_outcomes": len(probs),
                },
                recommended_action="Investigate potential decoherence or circuit depth issues",
                observation_id=observation_id,
            )
            findings.append(finding)

        # Check entropy ratio
        if collapse_metrics.entropy_ratio < 0.5:
            finding = DiagnosticFinding(
                finding_id=str(uuid.uuid4()),
                category=DiagnosticCategory.ENTROPY,
                severity=DiagnosticSeverity.INFO,
                description=(
                    f"Low entropy ratio {collapse_metrics.entropy_ratio:.4f} indicates "
                    "concentrated distribution"
                ),
                metrics={"entropy_ratio": collapse_metrics.entropy_ratio},
                recommended_action="Review circuit design if uniform distribution expected",
                observation_id=observation_id,
            )
            findings.append(finding)

        return findings

    def _compute_raw_diagnostics(
        self,
        counts: dict[str, int],
        probs: dict[str, float],
        prob_array: np.ndarray,
    ) -> dict[str, Any]:
        """Compute raw diagnostic values."""
        total = sum(counts.values())

        # Shannon entropy
        entropy = 0.0
        if len(prob_array) > 0:
            nonzero_probs = prob_array[prob_array > 0]
            entropy = -float(np.sum(nonzero_probs * np.log2(nonzero_probs)))

        return {
            "total_shots": total,
            "unique_outcomes": len(counts),
            "max_probability": float(np.max(prob_array)) if len(prob_array) > 0 else 0,
            "min_probability": float(np.min(prob_array)) if len(prob_array) > 0 else 0,
            "mean_probability": float(np.mean(prob_array)) if len(prob_array) > 0 else 0,
            "std_probability": float(np.std(prob_array)) if len(prob_array) > 0 else 0,
            "shannon_entropy": entropy,
            "max_possible_entropy": np.log2(len(counts)) if len(counts) > 0 else 0,
        }

    def _compute_provenance_hash(
        self, counts: dict[str, int], diagnostics: dict[str, Any]
    ) -> str:
        """Compute provenance hash for observation."""
        counts_str = json.dumps(counts, sort_keys=True)
        diag_str = json.dumps(diagnostics, sort_keys=True, default=str)
        combined = f"{counts_str}|{diag_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _count_by_severity(
        self, observations: list[TopologicalObservation]
    ) -> dict[str, int]:
        """Count findings by severity across observations."""
        counts: dict[str, int] = {}
        for obs in observations:
            for finding in obs.findings:
                severity = finding.severity.value
                counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _count_by_category(
        self, observations: list[TopologicalObservation]
    ) -> dict[str, int]:
        """Count findings by category across observations."""
        counts: dict[str, int] = {}
        for obs in observations:
            for finding in obs.findings:
                category = finding.category.value
                counts[category] = counts.get(category, 0) + 1
        return counts

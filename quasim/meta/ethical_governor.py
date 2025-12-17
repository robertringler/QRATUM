"""Quantum Ethical Governor (QEG) for QuASIM Phase VIII.

Implements supervisory agent for maintaining energy-equity balance constraints,
monitoring resource usage, and computing ethical compliance scores.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from quasim.audit.log import audit_event


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    energy_consumption: float  # kWh
    compute_time: float  # seconds
    memory_usage: float  # GB
    network_bandwidth: float  # Mbps
    timestamp: str


@dataclass
class FairnessMetrics:
    """Fairness and equity metrics."""

    gini_coefficient: float  # 0 (perfect equality) to 1 (perfect inequality)
    access_equity_score: float  # 0-100
    resource_distribution_score: float  # 0-100
    priority_fairness: float  # 0-100


@dataclass
class EthicalAssessment:
    """Ethical compliance assessment result."""

    ethics_score: float  # 0-100
    energy_efficiency: float  # 0-100
    equity_balance: float  # 0-100
    sustainability_score: float  # 0-100
    violations: List[str]
    recommendations: List[str]
    timestamp: str


class QuantumEthicalGovernor:
    """Supervisory agent for ethical governance.

    The QEG monitors resource usage, enforces energy-equity constraints,
    and emits ethical compliance scores to the DVL (Digital Verification Ledger).
    """

    def __init__(
        self,
        energy_budget: float = 1000.0,  # kWh per day
        equity_threshold: float = 0.3,  # Max Gini coefficient
        min_sustainability_score: float = 75.0,
    ):
        """Initialize the Quantum Ethical Governor.

        Parameters
        ----------
        energy_budget : float
            Daily energy budget in kWh
        equity_threshold : float
            Maximum acceptable Gini coefficient (0-1)
        min_sustainability_score : float
            Minimum required sustainability score (0-100)
        """

        self.energy_budget = energy_budget
        self.equity_threshold = equity_threshold
        self.min_sustainability_score = min_sustainability_score

        # Historical metrics
        self.resource_history: List[ResourceMetrics] = []
        self.fairness_history: List[FairnessMetrics] = []
        self.assessment_history: List[EthicalAssessment] = []

    def monitor_resources(
        self,
        energy_consumption: float,
        compute_time: float,
        memory_usage: float,
        network_bandwidth: float,
    ) -> ResourceMetrics:
        """Monitor resource usage.

        Parameters
        ----------
        energy_consumption : float
            Energy consumption in kWh
        compute_time : float
            Compute time in seconds
        memory_usage : float
            Memory usage in GB
        network_bandwidth : float
            Network bandwidth in Mbps

        Returns
        -------
        ResourceMetrics
            Recorded resource metrics
        """

        metrics = ResourceMetrics(
            energy_consumption=energy_consumption,
            compute_time=compute_time,
            memory_usage=memory_usage,
            network_bandwidth=network_bandwidth,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        )

        self.resource_history.append(metrics)

        # Log resource monitoring
        audit_event(
            "qeg.resources_monitored",
            {
                "energy_consumption": energy_consumption,
                "compute_time": compute_time,
                "memory_usage": memory_usage,
                "network_bandwidth": network_bandwidth,
            },
        )

        return metrics

    def assess_fairness(
        self,
        resource_distribution: List[float],
        access_counts: List[int],
        priority_levels: List[int],
    ) -> FairnessMetrics:
        """Assess fairness and equity of resource distribution.

        Parameters
        ----------
        resource_distribution : List[float]
            Resource allocation per user/tenant
        access_counts : List[int]
            Access counts per user/tenant
        priority_levels : List[int]
            Priority levels assigned to users/tenants

        Returns
        -------
        FairnessMetrics
            Fairness assessment metrics
        """

        # Calculate Gini coefficient
        gini = self._calculate_gini(resource_distribution)

        # Calculate access equity
        access_equity = self._calculate_access_equity(access_counts)

        # Calculate resource distribution score
        resource_dist_score = (1.0 - gini) * 100.0

        # Calculate priority fairness
        priority_fairness = self._calculate_priority_fairness(
            priority_levels, resource_distribution
        )

        metrics = FairnessMetrics(
            gini_coefficient=gini,
            access_equity_score=access_equity,
            resource_distribution_score=resource_dist_score,
            priority_fairness=priority_fairness,
        )

        self.fairness_history.append(metrics)

        # Log fairness assessment
        audit_event(
            "qeg.fairness_assessed",
            {
                "gini_coefficient": gini,
                "access_equity_score": access_equity,
                "resource_distribution_score": resource_dist_score,
                "priority_fairness": priority_fairness,
            },
        )

        return metrics

    def compute_ethical_score(
        self,
        resource_metrics: Optional[ResourceMetrics] = None,
        fairness_metrics: Optional[FairnessMetrics] = None,
    ) -> EthicalAssessment:
        """Compute comprehensive ethical compliance score.

        Parameters
        ----------
        resource_metrics : Optional[ResourceMetrics]
            Current resource metrics (uses latest if None)
        fairness_metrics : Optional[FairnessMetrics]
            Current fairness metrics (uses latest if None)

        Returns
        -------
        EthicalAssessment
            Ethical compliance assessment
        """

        # Use latest metrics if not provided
        if resource_metrics is None and self.resource_history:
            resource_metrics = self.resource_history[-1]
        if fairness_metrics is None and self.fairness_history:
            fairness_metrics = self.fairness_history[-1]

        violations = []
        recommendations = []

        # Energy efficiency assessment
        energy_efficiency = 100.0
        if resource_metrics:
            energy_usage_ratio = resource_metrics.energy_consumption / self.energy_budget
            if energy_usage_ratio > 1.0:
                violations.append(
                    f"Energy budget exceeded by {(energy_usage_ratio - 1.0) * 100:.1f}%"
                )
                energy_efficiency = max(0.0, 100.0 - (energy_usage_ratio - 1.0) * 100.0)
                recommendations.append("Reduce energy consumption or increase budget")
            else:
                energy_efficiency = 100.0 - (energy_usage_ratio * 20.0)

        # Equity balance assessment
        equity_balance = 100.0
        if fairness_metrics:
            if fairness_metrics.gini_coefficient > self.equity_threshold:
                violations.append(
                    f"Gini coefficient {fairness_metrics.gini_coefficient:.2f} exceeds threshold {self.equity_threshold}"
                )
                equity_balance = max(
                    0.0,
                    100.0
                    - (
                        (fairness_metrics.gini_coefficient - self.equity_threshold)
                        / (1.0 - self.equity_threshold)
                    )
                    * 100.0,
                )
                recommendations.append("Improve resource distribution equity")
            else:
                equity_balance = fairness_metrics.resource_distribution_score

        # Sustainability assessment
        sustainability_score = (energy_efficiency + equity_balance) / 2.0

        if sustainability_score < self.min_sustainability_score:
            violations.append(
                f"Sustainability score {sustainability_score:.1f} below minimum {self.min_sustainability_score}"
            )
            recommendations.append("Implement sustainability improvements")

        # Overall ethics score (weighted average)
        ethics_score = energy_efficiency * 0.4 + equity_balance * 0.4 + sustainability_score * 0.2

        assessment = EthicalAssessment(
            ethics_score=ethics_score,
            energy_efficiency=energy_efficiency,
            equity_balance=equity_balance,
            sustainability_score=sustainability_score,
            violations=violations,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        )

        self.assessment_history.append(assessment)

        # Log ethical assessment
        audit_event(
            "qeg.ethical_score_computed",
            {
                "ethics_score": ethics_score,
                "energy_efficiency": energy_efficiency,
                "equity_balance": equity_balance,
                "sustainability_score": sustainability_score,
                "violations_count": len(violations),
            },
        )

        return assessment

    def emit_to_dvl(self, assessment: EthicalAssessment) -> Dict[str, Any]:
        """Emit ethical compliance score to Digital Verification Ledger.

        Parameters
        ----------
        assessment : EthicalAssessment
            Ethical assessment to emit

        Returns
        -------
        Dict[str, Any]
            DVL emission record
        """

        dvl_record = {
            "record_type": "ethical_compliance",
            "timestamp": assessment.timestamp,
            "ethics_score": assessment.ethics_score,
            "energy_efficiency": assessment.energy_efficiency,
            "equity_balance": assessment.equity_balance,
            "sustainability_score": assessment.sustainability_score,
            "violations": assessment.violations,
            "recommendations": assessment.recommendations,
            "attestation": "QEG-v1.0.0",
        }

        # Log DVL emission
        audit_event(
            "qeg.dvl_emission",
            dvl_record,
        )

        return dvl_record

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of QEG operations.

        Returns
        -------
        Dict[str, Any]
            Performance summary
        """

        if not self.assessment_history:
            return {
                "assessments_count": 0,
                "avg_ethics_score": 0.0,
                "violations_count": 0,
            }

        avg_ethics_score = sum(a.ethics_score for a in self.assessment_history) / len(
            self.assessment_history
        )
        total_violations = sum(len(a.violations) for a in self.assessment_history)

        latest_assessment = self.assessment_history[-1] if self.assessment_history else None

        return {
            "assessments_count": len(self.assessment_history),
            "avg_ethics_score": avg_ethics_score,
            "latest_ethics_score": (latest_assessment.ethics_score if latest_assessment else 0.0),
            "violations_count": total_violations,
            "energy_budget": self.energy_budget,
            "equity_threshold": self.equity_threshold,
            "min_sustainability_score": self.min_sustainability_score,
        }

    def _calculate_gini(self, distribution: List[float]) -> float:
        """Calculate Gini coefficient for resource distribution.

        Parameters
        ----------
        distribution : List[float]
            Resource distribution values

        Returns
        -------
        float
            Gini coefficient (0-1)
        """

        if not distribution or len(distribution) < 2:
            return 0.0

        # Sort distribution
        sorted_dist = sorted(distribution)
        n = len(sorted_dist)
        cumsum = sum((i + 1) * val for i, val in enumerate(sorted_dist))
        total = sum(sorted_dist)

        if total == 0:
            return 0.0

        return (2 * cumsum) / (n * total) - (n + 1) / n

    def _calculate_access_equity(self, access_counts: List[int]) -> float:
        """Calculate access equity score.

        Parameters
        ----------
        access_counts : List[int]
            Access counts per user

        Returns
        -------
        float
            Access equity score (0-100)
        """

        if not access_counts:
            return 100.0

        mean_access = sum(access_counts) / len(access_counts)
        if mean_access == 0:
            return 100.0

        variance = sum((c - mean_access) ** 2 for c in access_counts) / len(access_counts)
        std_dev = variance**0.5
        coefficient_of_variation = std_dev / mean_access

        # Convert to score (lower CV = higher equity)
        equity_score = max(0.0, 100.0 - (coefficient_of_variation * 50.0))
        return equity_score

    def _calculate_priority_fairness(
        self, priority_levels: List[int], resource_distribution: List[float]
    ) -> float:
        """Calculate priority fairness score.

        Parameters
        ----------
        priority_levels : List[int]
            Priority levels assigned
        resource_distribution : List[float]
            Resource allocations

        Returns
        -------
        float
            Priority fairness score (0-100)
        """

        if not priority_levels or not resource_distribution:
            return 100.0

        if len(priority_levels) != len(resource_distribution):
            return 50.0

        # Check if higher priority correlates with more resources
        # (in reasonable proportion)
        total_priority = sum(priority_levels)
        total_resources = sum(resource_distribution)

        if total_priority == 0 or total_resources == 0:
            return 100.0

        misalignment = 0.0
        for priority, resource in zip(priority_levels, resource_distribution):
            expected_ratio = priority / total_priority
            actual_ratio = resource / total_resources
            misalignment += abs(expected_ratio - actual_ratio)

        # Convert misalignment to score
        fairness_score = max(0.0, 100.0 - (misalignment * 100.0))
        return fairness_score

"""QRATUM Metrics Module.

This module implements the metrics required by the QRATUM ASCENSION DIRECTIVE.
All modules must report these metrics to be considered valid:

- Outcome Superiority Ratio (OSR): Measures outcome quality relative to baseline
- Compute Efficiency Index (CEI): Outcome per unit of compute cost
- Sovereignty Factor (SF): Degree of independence from external systems (0-1)
- Hallucination Risk Density (HRD): Risk of non-deterministic/incorrect outputs (~0)

A module is INVALID unless it reports all four metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from enum import Enum


class MetricStatus(Enum):
    """Status of a metric measurement."""

    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    NOT_MEASURED = "not_measured"


@dataclass
class MetricMeasurement:
    """A single metric measurement with metadata."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    status: MetricStatus = MetricStatus.VALID
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if measurement is valid."""
        return self.status == MetricStatus.VALID and self.value >= 0


@dataclass
class QRATUMMetrics:
    """QRATUM directive-mandated metrics.

    All modules must report these four metrics.
    """

    # Outcome Superiority Ratio (OSR)
    # Measures how much better outcomes are vs baseline
    # OSR = actual_outcome / baseline_outcome
    # Target: > 1.0 (better than baseline)
    outcome_superiority_ratio: float = 0.0

    # Compute Efficiency Index (CEI)
    # Measures outcome per unit of compute
    # CEI = outcome_superiority / (compute_cost * external_dependence)
    # Target: Maximize
    compute_efficiency_index: float = 0.0

    # Sovereignty Factor (SF)
    # Degree of independence from external systems
    # SF = 1.0 - (external_calls / total_operations)
    # Target: 1.0 (fully sovereign/local)
    sovereignty_factor: float = 1.0

    # Hallucination Risk Density (HRD)
    # Risk of non-deterministic or incorrect outputs
    # HRD = non_deterministic_operations / total_operations
    # Target: ~0 (deterministic)
    hallucination_risk_density: float = 0.0

    # Tracking fields
    _total_operations: int = field(default=0, repr=False)
    _external_operations: int = field(default=0, repr=False)
    _non_deterministic_ops: int = field(default=0, repr=False)
    _baseline_outcome: float = field(default=1.0, repr=False)
    _actual_outcome: float = field(default=1.0, repr=False)
    _compute_cost: float = field(default=1.0, repr=False)

    def is_valid(self) -> bool:
        """Check if all metrics are valid per QRATUM directive."""
        return (
            self.outcome_superiority_ratio >= 0.0
            and self.compute_efficiency_index >= 0.0
            and 0.0 <= self.sovereignty_factor <= 1.0
            and self.hallucination_risk_density >= 0.0
            and self.hallucination_risk_density <= 1.0
        )

    def record_operation(
        self,
        external: bool = False,
        deterministic: bool = True,
        compute_units: float = 1.0,
    ) -> None:
        """Record an operation for metric calculation.

        Args:
            external: Whether operation depends on external system.
            deterministic: Whether operation is deterministic.
            compute_units: Relative compute cost of operation.
        """
        self._total_operations += 1
        self._compute_cost += compute_units

        if external:
            self._external_operations += 1

        if not deterministic:
            self._non_deterministic_ops += 1

        self._update_metrics()

    def record_outcome(self, actual: float, baseline: float = 1.0) -> None:
        """Record an outcome for OSR calculation.

        Args:
            actual: Actual outcome value.
            baseline: Baseline outcome for comparison.
        """
        self._actual_outcome = actual
        self._baseline_outcome = baseline
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update all metrics based on recorded data."""
        # OSR
        if self._baseline_outcome > 0:
            self.outcome_superiority_ratio = self._actual_outcome / self._baseline_outcome

        # SF
        if self._total_operations > 0:
            self.sovereignty_factor = 1.0 - (
                self._external_operations / self._total_operations
            )
        else:
            self.sovereignty_factor = 1.0

        # HRD
        if self._total_operations > 0:
            self.hallucination_risk_density = (
                self._non_deterministic_ops / self._total_operations
            )
        else:
            self.hallucination_risk_density = 0.0

        # CEI = outcome / (compute * external_dependence)
        external_factor = 1.0 + (1.0 - self.sovereignty_factor)
        if self._compute_cost > 0 and external_factor > 0:
            self.compute_efficiency_index = (
                self.outcome_superiority_ratio / (self._compute_cost * external_factor)
            )

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "OSR": self.outcome_superiority_ratio,
            "CEI": self.compute_efficiency_index,
            "SF": self.sovereignty_factor,
            "HRD": self.hallucination_risk_density,
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        return (
            f"QRATUMMetrics("
            f"OSR={self.outcome_superiority_ratio:.4f}, "
            f"CEI={self.compute_efficiency_index:.4f}, "
            f"SF={self.sovereignty_factor:.4f}, "
            f"HRD={self.hallucination_risk_density:.6f})"
        )


@runtime_checkable
class MetricsReporter(Protocol):
    """Protocol for modules that report QRATUM metrics."""

    def get_metrics(self) -> QRATUMMetrics:
        """Return current metrics."""
        ...


class MetricsAggregator:
    """Aggregates metrics from multiple modules."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._modules: dict[str, MetricsReporter] = {}
        self._history: list[tuple[float, str, QRATUMMetrics]] = []

    def register_module(self, name: str, module: MetricsReporter) -> None:
        """Register a module for metrics tracking.

        Args:
            name: Module identifier.
            module: Module implementing MetricsReporter.
        """
        self._modules[name] = module

    def collect(self) -> dict[str, QRATUMMetrics]:
        """Collect metrics from all registered modules.

        Returns:
            Dictionary mapping module names to their metrics.
        """
        results = {}
        timestamp = time.time()

        for name, module in self._modules.items():
            metrics = module.get_metrics()
            results[name] = metrics
            self._history.append((timestamp, name, metrics))

        return results

    def aggregate(self) -> QRATUMMetrics:
        """Compute aggregate metrics across all modules.

        Returns:
            Weighted average of all module metrics.
        """
        all_metrics = self.collect()

        if not all_metrics:
            return QRATUMMetrics()

        n = len(all_metrics)
        aggregated = QRATUMMetrics()

        aggregated.outcome_superiority_ratio = sum(
            m.outcome_superiority_ratio for m in all_metrics.values()
        ) / n
        aggregated.compute_efficiency_index = sum(
            m.compute_efficiency_index for m in all_metrics.values()
        ) / n
        aggregated.sovereignty_factor = min(
            m.sovereignty_factor for m in all_metrics.values()
        )  # Weakest link
        aggregated.hallucination_risk_density = max(
            m.hallucination_risk_density for m in all_metrics.values()
        )  # Worst case

        return aggregated

    def validate_all(self) -> dict[str, bool]:
        """Validate metrics from all modules.

        Returns:
            Dictionary mapping module names to validity status.
        """
        all_metrics = self.collect()
        return {name: metrics.is_valid() for name, metrics in all_metrics.items()}

    def get_invalid_modules(self) -> list[str]:
        """Get list of modules with invalid metrics.

        Returns:
            List of module names that fail validation.
        """
        validity = self.validate_all()
        return [name for name, valid in validity.items() if not valid]


def validate_module_metrics(module: Any) -> tuple[bool, str]:
    """Validate that a module properly reports QRATUM metrics.

    Args:
        module: Module to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check if module implements MetricsReporter
    if not hasattr(module, "get_metrics"):
        return False, "Module missing get_metrics() method"

    try:
        metrics = module.get_metrics()
    except Exception as e:
        return False, f"get_metrics() raised exception: {e}"

    if not isinstance(metrics, QRATUMMetrics):
        return False, "get_metrics() must return QRATUMMetrics instance"

    if not metrics.is_valid():
        return False, f"Metrics validation failed: {metrics}"

    return True, "Module metrics valid"

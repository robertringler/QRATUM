"""
Data Quality and Validation Module

Provides utilities for data quality assessment and validation
of SEER-Medicare pipeline inputs and outputs.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Generator, Optional

from .privacy import PrivacyConfig, SafeLogger, suppress_small_counts
from .schema import ClaimEvent, PatientTimeline, RegistryCase

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Data quality metrics.

    Attributes:
        total_records: Total records processed
        valid_records: Records passing validation
        invalid_records: Records failing validation
        error_counts: Counts by error type
        field_completeness: Completeness rate by field (0-1)
        date_range: Min and max dates observed
        warnings: List of warning messages
    """

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)
    field_completeness: dict[str, float] = field(default_factory=dict)
    date_range: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, min_cell_size: int = 11) -> dict[str, Any]:
        """Serialize with suppression."""
        valid = self.valid_records
        invalid = self.invalid_records
        suppressed = f"<{min_cell_size}"
        return {
            "total_records": self.total_records,
            "valid_records": valid if valid >= min_cell_size else suppressed,
            "invalid_records": invalid if invalid >= min_cell_size else suppressed,
            "error_counts": suppress_small_counts(self.error_counts, min_cell_size),
            "field_completeness": self.field_completeness,
            "date_range": self.date_range,
            "warnings": self.warnings,
        }


class DataValidator:
    """
    Validates data quality for SEER-Medicare pipeline.

    Checks for:
    - Required fields present
    - Valid date ranges
    - Consistent patient keys
    - Reasonable value ranges
    """

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize validator."""
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("data_validator", self.privacy_config)
        self._metrics = QualityMetrics()

    def validate_registry_case(self, case: RegistryCase) -> list[str]:
        """Validate a registry case.

        Args:
            case: RegistryCase to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not case.patient_key:
            errors.append("missing_patient_key")
        if not case.dx_date:
            errors.append("missing_dx_date")
        if not case.cancer_site:
            errors.append("missing_cancer_site")

        # Date validity
        if case.dx_date:
            if case.dx_date.year < 1970 or case.dx_date.year > 2030:
                errors.append("invalid_dx_date_range")

        if case.death_date:
            if case.dx_date and case.death_date < case.dx_date:
                errors.append("death_before_diagnosis")

        # Age validity
        if case.age_at_dx is not None:
            if case.age_at_dx < 0 or case.age_at_dx > 120:
                errors.append("invalid_age")

        # Survival consistency
        if case.survival_months is not None:
            if case.survival_months < 0:
                errors.append("negative_survival")

        return errors

    def validate_claim_event(self, event: ClaimEvent) -> list[str]:
        """Validate a claim event.

        Args:
            event: ClaimEvent to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not event.patient_key:
            errors.append("missing_patient_key")
        if not event.event_date:
            errors.append("missing_event_date")
        if not event.code:
            errors.append("missing_code")

        # Date validity
        if event.event_date:
            if event.event_date.year < 1999 or event.event_date.year > 2030:
                errors.append("invalid_event_date_range")

        # Cost validity
        if event.cost is not None:
            if event.cost < 0:
                errors.append("negative_cost")

        return errors

    def validate_timeline(self, timeline: PatientTimeline) -> list[str]:
        """Validate a patient timeline.

        Args:
            timeline: PatientTimeline to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not timeline.patient_key:
            errors.append("missing_patient_key")

        # Date consistency
        if timeline.lookback_start and timeline.followup_end:
            if timeline.lookback_start > timeline.followup_end:
                errors.append("invalid_window_dates")

        if timeline.index_date and timeline.lookback_start:
            if timeline.index_date < timeline.lookback_start:
                errors.append("index_before_lookback")

        # Treatment events consistency
        for event in timeline.treatment_events:
            if event.patient_key != timeline.patient_key:
                errors.append("inconsistent_patient_key")
                break

        return errors

    def compute_registry_quality_metrics(
        self,
        cases: Generator[RegistryCase, None, None],
    ) -> QualityMetrics:
        """Compute quality metrics for registry cases.

        Args:
            cases: Generator of RegistryCase objects

        Returns:
            QualityMetrics for the registry data
        """
        metrics = QualityMetrics()
        min_date: Optional[date] = None
        max_date: Optional[date] = None

        # Field counters
        field_totals = {
            "patient_key": 0,
            "dx_date": 0,
            "cancer_site": 0,
            "histology": 0,
            "stage": 0,
            "age_at_dx": 0,
            "sex": 0,
            "vital_status": 0,
            "survival_months": 0,
        }

        for case in cases:
            metrics.total_records += 1
            errors = self.validate_registry_case(case)

            if errors:
                metrics.invalid_records += 1
                for error in errors:
                    metrics.error_counts[error] = metrics.error_counts.get(error, 0) + 1
            else:
                metrics.valid_records += 1

            # Track completeness
            if case.patient_key:
                field_totals["patient_key"] += 1
            if case.dx_date:
                field_totals["dx_date"] += 1
                if min_date is None or case.dx_date < min_date:
                    min_date = case.dx_date
                if max_date is None or case.dx_date > max_date:
                    max_date = case.dx_date
            if case.cancer_site:
                field_totals["cancer_site"] += 1
            if case.histology:
                field_totals["histology"] += 1
            if case.stage:
                field_totals["stage"] += 1
            if case.age_at_dx is not None:
                field_totals["age_at_dx"] += 1
            if case.sex:
                field_totals["sex"] += 1
            if case.survival_months is not None:
                field_totals["survival_months"] += 1

        # Compute completeness rates
        if metrics.total_records > 0:
            for field_name, count in field_totals.items():
                metrics.field_completeness[field_name] = count / metrics.total_records

        # Date range
        if min_date:
            metrics.date_range["min_dx_date"] = min_date.isoformat()
        if max_date:
            metrics.date_range["max_dx_date"] = max_date.isoformat()

        # Warnings
        if metrics.invalid_records > metrics.total_records * 0.1:
            metrics.warnings.append("High error rate (>10%)")
        if metrics.field_completeness.get("stage", 1.0) < 0.5:
            metrics.warnings.append("Low stage completeness (<50%)")

        self._metrics = metrics
        return metrics

    def compute_claims_quality_metrics(
        self,
        events: Generator[ClaimEvent, None, None],
        sample_size: int = 100000,
    ) -> QualityMetrics:
        """Compute quality metrics for claims events.

        Args:
            events: Generator of ClaimEvent objects
            sample_size: Maximum events to process

        Returns:
            QualityMetrics for the claims data
        """
        metrics = QualityMetrics()
        min_date: Optional[date] = None
        max_date: Optional[date] = None

        field_totals = {
            "patient_key": 0,
            "event_date": 0,
            "code": 0,
            "cost": 0,
            "provider_type": 0,
        }

        count = 0
        for event in events:
            if count >= sample_size:
                metrics.warnings.append(f"Sampled first {sample_size} records")
                break

            metrics.total_records += 1
            count += 1

            errors = self.validate_claim_event(event)

            if errors:
                metrics.invalid_records += 1
                for error in errors:
                    metrics.error_counts[error] = metrics.error_counts.get(error, 0) + 1
            else:
                metrics.valid_records += 1

            # Track completeness
            if event.patient_key:
                field_totals["patient_key"] += 1
            if event.event_date:
                field_totals["event_date"] += 1
                if min_date is None or event.event_date < min_date:
                    min_date = event.event_date
                if max_date is None or event.event_date > max_date:
                    max_date = event.event_date
            if event.code:
                field_totals["code"] += 1
            if event.cost is not None:
                field_totals["cost"] += 1
            if event.provider_type:
                field_totals["provider_type"] += 1

        # Compute completeness rates
        if metrics.total_records > 0:
            for field_name, cnt in field_totals.items():
                metrics.field_completeness[field_name] = cnt / metrics.total_records

        # Date range
        if min_date:
            metrics.date_range["min_event_date"] = min_date.isoformat()
        if max_date:
            metrics.date_range["max_event_date"] = max_date.isoformat()

        self._metrics = metrics
        return metrics

    def get_metrics(self) -> QualityMetrics:
        """Get the last computed metrics."""
        return self._metrics

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary with suppression."""
        return self._metrics.to_dict(min_cell_size=self.privacy_config.min_cell_size)


def check_date_overlap(
    seer_date_range: dict[str, str],
    claims_date_range: dict[str, str],
) -> dict[str, Any]:
    """Check for overlap between SEER and claims date ranges.

    Args:
        seer_date_range: Date range from SEER data
        claims_date_range: Date range from claims data

    Returns:
        Dictionary with overlap assessment
    """
    result = {
        "has_overlap": False,
        "seer_range": seer_date_range,
        "claims_range": claims_date_range,
        "overlap_start": None,
        "overlap_end": None,
        "warnings": [],
    }

    try:
        seer_min = date.fromisoformat(seer_date_range.get("min_dx_date", ""))
        seer_max = date.fromisoformat(seer_date_range.get("max_dx_date", ""))
        claims_min = date.fromisoformat(claims_date_range.get("min_event_date", ""))
        claims_max = date.fromisoformat(claims_date_range.get("max_event_date", ""))

        overlap_start = max(seer_min, claims_min)
        overlap_end = min(seer_max, claims_max)

        if overlap_start <= overlap_end:
            result["has_overlap"] = True
            result["overlap_start"] = overlap_start.isoformat()
            result["overlap_end"] = overlap_end.isoformat()
        else:
            result["warnings"].append("No date overlap between SEER and claims")

    except (ValueError, TypeError):
        result["warnings"].append("Could not parse date ranges")

    return result


def generate_quality_report(
    registry_metrics: QualityMetrics,
    claims_metrics: Optional[QualityMetrics] = None,
    min_cell_size: int = 11,
) -> str:
    """Generate a quality report as Markdown text.

    Args:
        registry_metrics: Registry data quality metrics
        claims_metrics: Claims data quality metrics (optional)
        min_cell_size: Minimum cell size for suppression

    Returns:
        Markdown formatted quality report
    """
    lines = []
    lines.append("# Data Quality Report\n")

    # Registry section
    lines.append("## Registry Data Quality\n")
    lines.append(f"- Total records: {registry_metrics.total_records}")

    if registry_metrics.valid_records >= min_cell_size:
        lines.append(f"- Valid records: {registry_metrics.valid_records}")
    else:
        lines.append(f"- Valid records: <{min_cell_size} (suppressed)")

    lines.append("\n### Field Completeness\n")
    for field_name, rate in registry_metrics.field_completeness.items():
        lines.append(f"- {field_name}: {rate*100:.1f}%")

    if registry_metrics.date_range:
        lines.append("\n### Date Range\n")
        for key, val in registry_metrics.date_range.items():
            lines.append(f"- {key}: {val}")

    if registry_metrics.warnings:
        lines.append("\n### Warnings\n")
        for warning in registry_metrics.warnings:
            lines.append(f"- ⚠️ {warning}")

    # Claims section
    if claims_metrics:
        lines.append("\n## Claims Data Quality\n")
        lines.append(f"- Total records processed: {claims_metrics.total_records}")

        if claims_metrics.valid_records >= min_cell_size:
            lines.append(f"- Valid records: {claims_metrics.valid_records}")
        else:
            lines.append(f"- Valid records: <{min_cell_size} (suppressed)")

        lines.append("\n### Field Completeness\n")
        for field_name, rate in claims_metrics.field_completeness.items():
            lines.append(f"- {field_name}: {rate*100:.1f}%")

        if claims_metrics.warnings:
            lines.append("\n### Warnings\n")
            for warning in claims_metrics.warnings:
                lines.append(f"- ⚠️ {warning}")

    return "\n".join(lines)

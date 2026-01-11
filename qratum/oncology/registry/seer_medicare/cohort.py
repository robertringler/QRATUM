"""
Cohort Building Module

Provides utilities for building patient cohorts from SEER registry
cases with inclusion/exclusion criteria.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from .linkages import LinkedPatient
from .privacy import PrivacyConfig, SafeLogger, suppress_small_counts
from .schema import CohortDefinition, RegistryCase

logger = logging.getLogger(__name__)


@dataclass
class CohortStats:
    """Cohort statistics for reporting.

    All counts are subject to suppression per privacy configuration.

    Attributes:
        total_considered: Total cases considered
        included: Number included in final cohort
        exclusion_reasons: Counts by exclusion reason
        age_distribution: Age bin distribution
        sex_distribution: Sex distribution
        stage_distribution: Stage distribution
        vital_status_distribution: Vital status distribution
        diagnosis_year_distribution: Diagnosis year distribution
        missingness: Counts of missing values by field
    """

    total_considered: int = 0
    included: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    age_distribution: dict[str, int] = field(default_factory=dict)
    sex_distribution: dict[str, int] = field(default_factory=dict)
    stage_distribution: dict[str, int] = field(default_factory=dict)
    vital_status_distribution: dict[str, int] = field(default_factory=dict)
    diagnosis_year_distribution: dict[str, int] = field(default_factory=dict)
    missingness: dict[str, int] = field(default_factory=dict)

    def to_dict(
        self,
        min_cell_size: int = 11,
    ) -> dict[str, Any]:
        """Serialize with suppression of small cell sizes.

        Args:
            min_cell_size: Minimum cell size for suppression

        Returns:
            Dictionary with suppressed counts
        """
        return {
            "total_considered": self.total_considered,
            "included": self.included if self.included >= min_cell_size else f"<{min_cell_size}",
            "exclusion_reasons": suppress_small_counts(self.exclusion_reasons, min_cell_size),
            "age_distribution": suppress_small_counts(self.age_distribution, min_cell_size),
            "sex_distribution": suppress_small_counts(self.sex_distribution, min_cell_size),
            "stage_distribution": suppress_small_counts(self.stage_distribution, min_cell_size),
            "vital_status_distribution": suppress_small_counts(
                self.vital_status_distribution, min_cell_size
            ),
            "diagnosis_year_distribution": suppress_small_counts(
                self.diagnosis_year_distribution, min_cell_size
            ),
            "missingness": suppress_small_counts(self.missingness, min_cell_size),
        }


class CohortBuilder:
    """
    Builds patient cohorts from SEER registry cases.

    Applies inclusion/exclusion criteria and generates summary statistics.

    Usage:
        definition = CohortDefinition(
            name="lung_adenocarcinoma",
            cancer_site="lung",
            histology="adenocarcinoma",
            diagnosis_year_min=2010,
            diagnosis_year_max=2018,
        )
        builder = CohortBuilder(definition, privacy_config)

        for patient in patients:
            if builder.check_inclusion(patient):
                cohort.append(patient)

        stats = builder.get_stats()
    """

    def __init__(
        self,
        definition: CohortDefinition,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize cohort builder.

        Args:
            definition: Cohort definition parameters
            privacy_config: Privacy configuration
        """
        self.definition = definition
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("cohort_builder", self.privacy_config)

        self._stats = CohortStats()
        self._included_patients: list[LinkedPatient] = []

        # Compile patterns for efficient matching
        self._site_patterns = self._compile_patterns(
            definition.cancer_site_patterns
            or ([definition.cancer_site] if definition.cancer_site else [])
        )
        self._histology_patterns = self._compile_patterns(
            definition.histology_patterns
            or ([definition.histology] if definition.histology else [])
        )
        self._stage_patterns = self._compile_patterns(
            definition.stage_patterns
            or ([definition.stage] if definition.stage else [])
        )

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern]:
        """Compile string patterns to regex."""
        compiled = []
        for p in patterns:
            if p:
                # Convert to regex, case-insensitive
                compiled.append(re.compile(re.escape(p), re.IGNORECASE))
        return compiled

    def _matches_patterns(self, value: str, patterns: list[re.Pattern]) -> bool:
        """Check if value matches any pattern."""
        if not patterns:
            return True  # No filter = match all
        if not value:
            return False
        for pattern in patterns:
            if pattern.search(value):
                return True
        return False

    def check_inclusion(
        self,
        patient: LinkedPatient,
        record_stats: bool = True,
    ) -> bool:
        """Check if patient meets inclusion criteria.

        Args:
            patient: LinkedPatient to check
            record_stats: Whether to record statistics

        Returns:
            True if patient should be included in cohort
        """
        if record_stats:
            self._stats.total_considered += 1

        case = patient.registry_case
        if case is None:
            if record_stats:
                self._stats.exclusion_reasons["no_registry_case"] = (
                    self._stats.exclusion_reasons.get("no_registry_case", 0) + 1
                )
            return False

        # Check cancer site
        if self._site_patterns:
            if not self._matches_patterns(case.cancer_site, self._site_patterns):
                if record_stats:
                    self._stats.exclusion_reasons["site_mismatch"] = (
                        self._stats.exclusion_reasons.get("site_mismatch", 0) + 1
                    )
                return False

        # Check histology
        if self._histology_patterns:
            if not self._matches_patterns(case.histology, self._histology_patterns):
                if record_stats:
                    self._stats.exclusion_reasons["histology_mismatch"] = (
                        self._stats.exclusion_reasons.get("histology_mismatch", 0) + 1
                    )
                return False

        # Check stage
        if self._stage_patterns:
            if not self._matches_patterns(case.stage, self._stage_patterns):
                if record_stats:
                    self._stats.exclusion_reasons["stage_mismatch"] = (
                        self._stats.exclusion_reasons.get("stage_mismatch", 0) + 1
                    )
                return False

        # Check diagnosis year
        if case.dx_date is None:
            if record_stats:
                self._stats.exclusion_reasons["missing_dx_date"] = (
                    self._stats.exclusion_reasons.get("missing_dx_date", 0) + 1
                )
            return False

        dx_year = case.dx_date.year
        if self.definition.diagnosis_year_min and dx_year < self.definition.diagnosis_year_min:
            if record_stats:
                self._stats.exclusion_reasons["dx_year_too_early"] = (
                    self._stats.exclusion_reasons.get("dx_year_too_early", 0) + 1
                )
            return False

        if self.definition.diagnosis_year_max and dx_year > self.definition.diagnosis_year_max:
            if record_stats:
                self._stats.exclusion_reasons["dx_year_too_late"] = (
                    self._stats.exclusion_reasons.get("dx_year_too_late", 0) + 1
                )
            return False

        # Check age
        age_min = self.definition.age_min
        if age_min and (case.age_at_dx is None or case.age_at_dx < age_min):
            if record_stats:
                self._stats.exclusion_reasons["age_too_young"] = (
                    self._stats.exclusion_reasons.get("age_too_young", 0) + 1
                )
            return False

        if self.definition.age_max and case.age_at_dx and case.age_at_dx > self.definition.age_max:
            if record_stats:
                self._stats.exclusion_reasons["age_too_old"] = (
                    self._stats.exclusion_reasons.get("age_too_old", 0) + 1
                )
            return False

        # Check sex
        if self.definition.sex:
            if case.sex.upper() != self.definition.sex.upper():
                if record_stats:
                    self._stats.exclusion_reasons["sex_mismatch"] = (
                        self._stats.exclusion_reasons.get("sex_mismatch", 0) + 1
                    )
                return False

        # Check continuous enrollment
        if self.definition.require_continuous_enrollment:
            if not patient.has_continuous_enrollment:
                if record_stats:
                    self._stats.exclusion_reasons["no_continuous_enrollment"] = (
                        self._stats.exclusion_reasons.get("no_continuous_enrollment", 0) + 1
                    )
                return False

        # Check prior cancer (sequence number)
        if self.definition.exclude_prior_cancer:
            if case.sequence_number > 0:
                if record_stats:
                    self._stats.exclusion_reasons["prior_cancer"] = (
                        self._stats.exclusion_reasons.get("prior_cancer", 0) + 1
                    )
                return False

        # Patient passed all criteria
        if record_stats:
            self._stats.included += 1
            self._record_distributions(case)

        return True

    def _record_distributions(self, case: RegistryCase) -> None:
        """Record distribution statistics for included patient."""
        # Age distribution
        if case.age_at_dx is not None:
            age_bin = self._get_age_bin(case.age_at_dx)
            self._stats.age_distribution[age_bin] = (
                self._stats.age_distribution.get(age_bin, 0) + 1
            )
        else:
            self._stats.missingness["age"] = self._stats.missingness.get("age", 0) + 1

        # Sex distribution
        sex = case.sex.upper() if case.sex else "Unknown"
        self._stats.sex_distribution[sex] = self._stats.sex_distribution.get(sex, 0) + 1

        # Stage distribution
        stage = case.stage if case.stage else "Unknown"
        self._stats.stage_distribution[stage] = self._stats.stage_distribution.get(stage, 0) + 1

        # Vital status
        vs = case.vital_status.value
        self._stats.vital_status_distribution[vs] = (
            self._stats.vital_status_distribution.get(vs, 0) + 1
        )

        # Diagnosis year
        if case.dx_date:
            year_str = str(case.dx_date.year)
            self._stats.diagnosis_year_distribution[year_str] = (
                self._stats.diagnosis_year_distribution.get(year_str, 0) + 1
            )

    def _get_age_bin(self, age: int, bin_size: int = 5) -> str:
        """Get age bin label."""
        if age < 65:
            return "<65"
        elif age >= 85:
            return "85+"
        else:
            bin_start = (age // bin_size) * bin_size
            bin_end = bin_start + bin_size - 1
            return f"{bin_start}-{bin_end}"

    def build_cohort(
        self,
        patients: Generator[LinkedPatient, None, None],
    ) -> list[LinkedPatient]:
        """Build cohort from patient stream.

        Args:
            patients: Generator of LinkedPatient objects

        Returns:
            List of patients meeting cohort criteria
        """
        self._included_patients = []

        for patient in patients:
            if self.check_inclusion(patient):
                self._included_patients.append(patient)

        self.safe_logger.log_cohort_summary(
            total=self._stats.total_considered,
            included=self._stats.included,
            excluded_reasons=self._stats.exclusion_reasons,
        )

        return self._included_patients

    def get_stats(self) -> CohortStats:
        """Get cohort statistics."""
        return self._stats

    def get_stats_dict(self) -> dict[str, Any]:
        """Get cohort statistics as dictionary with suppression."""
        return self._stats.to_dict(min_cell_size=self.privacy_config.min_cell_size)

    def get_included_patients(self) -> list[LinkedPatient]:
        """Get list of included patients."""
        return self._included_patients


def create_cohort_definition_lung_adenocarcinoma(
    diagnosis_year_min: int = 2010,
    diagnosis_year_max: int = 2018,
) -> CohortDefinition:
    """Create a standard lung adenocarcinoma cohort definition.

    Args:
        diagnosis_year_min: Minimum diagnosis year
        diagnosis_year_max: Maximum diagnosis year

    Returns:
        CohortDefinition for lung adenocarcinoma
    """
    return CohortDefinition(
        name="lung_adenocarcinoma",
        cancer_site="lung",
        cancer_site_patterns=["LUNG", "C34"],
        histology="adenocarcinoma",
        histology_patterns=["ADENOCARCINOMA", "8140", "8250", "8260", "8310", "8255"],
        diagnosis_year_min=diagnosis_year_min,
        diagnosis_year_max=diagnosis_year_max,
        age_min=65,  # Medicare population
        require_continuous_enrollment=True,
        enrollment_months_pre=12,
        enrollment_months_post=12,
        exclude_prior_cancer=True,
        metadata={"description": "Stage IV lung adenocarcinoma in Medicare population"},
    )


def create_cohort_definition_breast_cancer(
    diagnosis_year_min: int = 2010,
    diagnosis_year_max: int = 2018,
) -> CohortDefinition:
    """Create a standard breast cancer cohort definition."""
    return CohortDefinition(
        name="breast_cancer",
        cancer_site="breast",
        cancer_site_patterns=["BREAST", "C50"],
        diagnosis_year_min=diagnosis_year_min,
        diagnosis_year_max=diagnosis_year_max,
        age_min=65,
        sex="F",
        require_continuous_enrollment=True,
        exclude_prior_cancer=True,
        metadata={"description": "Female breast cancer in Medicare population"},
    )


def create_cohort_definition_colorectal_cancer(
    diagnosis_year_min: int = 2010,
    diagnosis_year_max: int = 2018,
) -> CohortDefinition:
    """Create a standard colorectal cancer cohort definition."""
    return CohortDefinition(
        name="colorectal_cancer",
        cancer_site="colon",
        cancer_site_patterns=["COLON", "RECTUM", "C18", "C19", "C20"],
        diagnosis_year_min=diagnosis_year_min,
        diagnosis_year_max=diagnosis_year_max,
        age_min=65,
        require_continuous_enrollment=True,
        exclude_prior_cancer=True,
        metadata={"description": "Colorectal cancer in Medicare population"},
    )

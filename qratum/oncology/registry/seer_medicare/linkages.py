"""
Patient Linkage and Eligibility Utilities

Provides utilities for linking SEER cases to Medicare claims
and verifying enrollment eligibility.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

from .privacy import PrivacyConfig, SafeLogger, create_patient_key
from .schema import ClaimEvent, RegistryCase

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentPeriod:
    """Medicare enrollment period for a patient.

    Attributes:
        patient_key: Hashed patient key
        start_date: Enrollment start date
        end_date: Enrollment end date
        part_a: Part A enrolled
        part_b: Part B enrolled
        hmo: HMO enrollment flag (affects claims completeness)
        part_d: Part D enrolled
    """

    patient_key: str
    start_date: date
    end_date: date
    part_a: bool = True
    part_b: bool = True
    hmo: bool = False
    part_d: bool = False

    def covers_date(self, check_date: date) -> bool:
        """Check if this period covers a specific date."""
        return self.start_date <= check_date <= self.end_date

    def covers_range(self, start: date, end: date) -> bool:
        """Check if this period fully covers a date range."""
        return self.start_date <= start and self.end_date >= end


@dataclass
class LinkedPatient:
    """Linked SEER-Medicare patient record.

    Attributes:
        patient_key: Hashed patient key
        registry_case: Associated SEER registry case
        enrollment_periods: List of enrollment periods
        has_continuous_enrollment: Whether continuous enrollment exists
        continuous_start: Start of continuous enrollment period
        continuous_end: End of continuous enrollment period
    """

    patient_key: str
    registry_case: Optional[RegistryCase] = None
    enrollment_periods: list[EnrollmentPeriod] = field(default_factory=list)
    has_continuous_enrollment: bool = False
    continuous_start: Optional[date] = None
    continuous_end: Optional[date] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def check_enrollment(
        self,
        start: date,
        end: date,
        require_part_a: bool = True,
        require_part_b: bool = True,
        exclude_hmo: bool = True,
    ) -> bool:
        """Check if patient has valid enrollment for a period.

        Args:
            start: Period start date
            end: Period end date
            require_part_a: Require Part A enrollment
            require_part_b: Require Part B enrollment
            exclude_hmo: Exclude HMO months

        Returns:
            True if enrollment criteria met
        """
        current = start
        while current <= end:
            has_coverage = False
            for period in self.enrollment_periods:
                if period.covers_date(current):
                    if require_part_a and not period.part_a:
                        continue
                    if require_part_b and not period.part_b:
                        continue
                    if exclude_hmo and period.hmo:
                        continue
                    has_coverage = True
                    break
            if not has_coverage:
                return False
            current += timedelta(days=1)
        return True


class PatientLinker:
    """
    Links SEER registry cases to Medicare claims data.

    Handles patient key mapping and enrollment verification.
    """

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize linker.

        Args:
            privacy_config: Privacy configuration
        """
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("linkage", self.privacy_config)

        self._registry_cases: dict[str, RegistryCase] = {}
        self._enrollment: dict[str, list[EnrollmentPeriod]] = {}
        self._linked_patients: dict[str, LinkedPatient] = {}

    def add_registry_case(self, case: RegistryCase) -> None:
        """Add a registry case.

        Args:
            case: RegistryCase to add
        """
        self._registry_cases[case.patient_key] = case

    def add_registry_cases(self, cases: Generator[RegistryCase, None, None]) -> int:
        """Add multiple registry cases.

        Args:
            cases: Generator of RegistryCase objects

        Returns:
            Number of cases added
        """
        count = 0
        for case in cases:
            self.add_registry_case(case)
            count += 1
        self.safe_logger.info_aggregate("Registry cases loaded", count)
        return count

    def load_enrollment_file(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> int:
        """Load enrollment data from file.

        Expected columns: BENE_ID, START_DATE, END_DATE, PART_A, PART_B, HMO, PART_D

        Args:
            filepath: Path to enrollment file
            encoding: File encoding
            delimiter: Field delimiter

        Returns:
            Number of enrollment records loaded
        """
        filepath = Path(filepath)
        self.safe_logger.info(f"Loading enrollment file: {filepath.name}")

        count = 0
        with open(filepath, encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                patient_id = row.get("BENE_ID") or row.get("PATIENT_ID") or row.get("bene_id")
                if not patient_id:
                    continue

                patient_key = create_patient_key(
                    patient_id,
                    salt=self.privacy_config.salt,
                    algorithm=self.privacy_config.hash_algorithm,
                )

                try:
                    start_str = row.get("START_DATE") or row.get("COVSTART")
                    end_str = row.get("END_DATE") or row.get("COVEND")

                    if not start_str or not end_str:
                        continue

                    # Parse dates
                    start_date = self._parse_date(start_str)
                    end_date = self._parse_date(end_str)

                    if not start_date or not end_date:
                        continue

                    # Parse flags
                    part_a = row.get("PART_A", "1") in ("1", "Y", "TRUE", "T")
                    part_b = row.get("PART_B", "1") in ("1", "Y", "TRUE", "T")
                    hmo = row.get("HMO", "0") in ("1", "Y", "TRUE", "T")
                    part_d = row.get("PART_D", "0") in ("1", "Y", "TRUE", "T")

                    period = EnrollmentPeriod(
                        patient_key=patient_key,
                        start_date=start_date,
                        end_date=end_date,
                        part_a=part_a,
                        part_b=part_b,
                        hmo=hmo,
                        part_d=part_d,
                    )

                    if patient_key not in self._enrollment:
                        self._enrollment[patient_key] = []
                    self._enrollment[patient_key].append(period)
                    count += 1

                except Exception:
                    continue

        self.safe_logger.info_aggregate("Enrollment records loaded", count)
        return count

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from string."""
        if not date_str:
            return None
        date_str = date_str.strip()
        try:
            if len(date_str) == 8 and date_str.isdigit():
                return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            if "-" in date_str:
                parts = date_str.split("-")
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass
        return None

    def build_linked_patients(self) -> dict[str, LinkedPatient]:
        """Build linked patient records from registry cases and enrollment.

        Returns:
            Dictionary of patient_key -> LinkedPatient
        """
        self._linked_patients = {}

        for patient_key, case in self._registry_cases.items():
            enrollment = self._enrollment.get(patient_key, [])

            linked = LinkedPatient(
                patient_key=patient_key,
                registry_case=case,
                enrollment_periods=enrollment,
            )

            # Check for continuous enrollment
            if case.dx_date and enrollment:
                linked.has_continuous_enrollment = linked.check_enrollment(
                    case.dx_date - timedelta(days=365),
                    case.dx_date + timedelta(days=365),
                )

            self._linked_patients[patient_key] = linked

        self.safe_logger.info_aggregate("Linked patients", len(self._linked_patients))
        return self._linked_patients

    def get_linked_patient(self, patient_key: str) -> Optional[LinkedPatient]:
        """Get a linked patient by key."""
        return self._linked_patients.get(patient_key)

    def get_patients_with_enrollment(
        self,
        pre_index_months: int = 12,
        post_index_months: int = 12,
    ) -> Generator[LinkedPatient, None, None]:
        """Get patients meeting enrollment requirements.

        Args:
            pre_index_months: Required months before index date
            post_index_months: Required months after index date

        Yields:
            LinkedPatient objects meeting criteria
        """
        pre_days = pre_index_months * 30
        post_days = post_index_months * 30

        for patient in self._linked_patients.values():
            if patient.registry_case is None:
                continue

            case = patient.registry_case
            if case.dx_date is None:
                continue

            start_date = case.dx_date - timedelta(days=pre_days)
            end_date = case.dx_date + timedelta(days=post_days)

            if patient.check_enrollment(start_date, end_date):
                patient.continuous_start = start_date
                patient.continuous_end = end_date
                patient.has_continuous_enrollment = True
                yield patient


def link_claims_to_patients(
    claims: Generator[ClaimEvent, None, None],
    linked_patients: dict[str, LinkedPatient],
) -> Generator[tuple[LinkedPatient, ClaimEvent], None, None]:
    """Link claims to their corresponding patients.

    Args:
        claims: Generator of ClaimEvent objects
        linked_patients: Dictionary of LinkedPatient objects

    Yields:
        Tuples of (LinkedPatient, ClaimEvent)
    """
    for claim in claims:
        patient = linked_patients.get(claim.patient_key)
        if patient:
            yield (patient, claim)

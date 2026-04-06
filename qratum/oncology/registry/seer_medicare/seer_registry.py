"""
SEER Registry Case File Parsing

Parses SEER cancer registry case files into normalized RegistryCase objects.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from .privacy import PrivacyConfig, SafeLogger, create_patient_key
from .schema import CauseOfDeath, RegistryCase, VitalStatus

logger = logging.getLogger(__name__)


# Common SEER field mappings (column names vary by data version)
SEER_FIELD_ALIASES = {
    "patient_id": ["PATIENT_ID", "patient_id", "PATIENTID", "patientid", "SEQ_NUM"],
    "dx_date": ["DATE_OF_DIAGNOSIS", "DX_DATE", "DT_DIAG", "DATEOFDIAGNOSIS"],
    "dx_year": ["YEAR_OF_DIAGNOSIS", "DX_YEAR", "YR_DIAG", "YEAROFDIAGNOSIS"],
    "dx_month": ["MONTH_OF_DIAGNOSIS", "DX_MONTH", "MO_DIAG"],
    "site": ["PRIMARY_SITE", "SITE", "PRIMARYSITE", "SITE_RECODE"],
    "histology": ["HISTOLOGY", "HIST", "HISTOLOGIC_TYPE", "HISTTYPE"],
    "stage": ["STAGE", "AJCC_STAGE", "SEER_STAGE", "COMBINED_STAGE", "STAGE_GROUP"],
    "grade": ["GRADE", "TUMOR_GRADE"],
    "laterality": ["LATERALITY", "LAT"],
    "age": ["AGE", "AGE_AT_DX", "AGE_DX", "AGERECODE"],
    "sex": ["SEX", "GENDER"],
    "race": ["RACE", "RACE1", "RACE_RECODE"],
    "vital_status": ["VITAL_STATUS", "VS", "STAT_REC"],
    "death_date": ["DATE_OF_DEATH", "DT_DEATH", "DEATHDATE"],
    "cause_of_death": ["COD", "CAUSE_OF_DEATH", "CAUSEOFDEATH", "COD_TO_SITE"],
    "survival_months": ["SURVIVAL", "SURV_MOS", "SURVIVAL_MONTHS"],
    "sequence_number": ["SEQUENCE_NUMBER", "SEQ_NUM", "SEQ", "SEQNUM"],
}


def _find_field(row: dict[str, str], aliases: list[str]) -> Optional[str]:
    """Find field value using multiple possible column names.

    Args:
        row: Row dictionary
        aliases: List of possible column names

    Returns:
        Field value or None if not found
    """
    for alias in aliases:
        if alias in row:
            return row[alias]
        # Try case-insensitive
        for key in row:
            if key.upper() == alias.upper():
                return row[key]
    return None


def _parse_date(date_str: Optional[str], year_str: Optional[str] = None) -> Optional[date]:
    """Parse date from various SEER formats.

    Args:
        date_str: Date string (various formats)
        year_str: Year string if date is split

    Returns:
        date object or None
    """
    if not date_str and not year_str:
        return None

    try:
        # Try YYYYMMDD format
        if date_str and len(date_str) == 8 and date_str.isdigit():
            return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))

        # Try YYYY-MM-DD format
        if date_str and "-" in date_str:
            parts = date_str.split("-")
            if len(parts) == 3:
                return date(int(parts[0]), int(parts[1]), int(parts[2]))

        # Try MM/DD/YYYY format
        if date_str and "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                return date(int(parts[2]), int(parts[0]), int(parts[1]))

        # If only year is available, use January 1
        if year_str and year_str.isdigit():
            return date(int(year_str), 1, 1)

    except (ValueError, IndexError):
        pass

    return None


def _parse_vital_status(status_str: Optional[str]) -> VitalStatus:
    """Parse vital status from SEER codes.

    Args:
        status_str: Vital status string

    Returns:
        VitalStatus enum value
    """
    if not status_str:
        return VitalStatus.UNKNOWN

    status_str = status_str.strip().upper()

    # Common SEER codes
    if status_str in ("1", "ALIVE", "A"):
        return VitalStatus.ALIVE
    elif status_str in ("4", "0", "DEAD", "D"):
        return VitalStatus.DEAD

    return VitalStatus.UNKNOWN


def _parse_cause_of_death(cod_str: Optional[str], cancer_site: str) -> CauseOfDeath:
    """Parse cause of death from SEER codes.

    Args:
        cod_str: Cause of death string
        cancer_site: Primary cancer site for comparison

    Returns:
        CauseOfDeath enum value
    """
    if not cod_str:
        return CauseOfDeath.UNKNOWN

    cod_str = cod_str.strip().upper()

    # SEER COD to site recode
    if cod_str in ("1", "CANCER", "MALIG"):
        return CauseOfDeath.CANCER
    elif cod_str in ("0", "7", "8", "9", "OTHER"):
        return CauseOfDeath.OTHER_CAUSE

    return CauseOfDeath.UNKNOWN


class SEERRegistryParser:
    """
    Parser for SEER cancer registry case files.

    Reads SEER case files (CSV format) and yields normalized RegistryCase objects.

    Usage:
        parser = SEERRegistryParser(privacy_config)
        for case in parser.parse_file(filepath):
            process(case)
    """

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize parser.

        Args:
            privacy_config: Privacy configuration for hashing
        """
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("seer_registry", self.privacy_config)

    def parse_file(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> Generator[RegistryCase, None, None]:
        """Parse a SEER registry file.

        Args:
            filepath: Path to SEER file
            encoding: File encoding
            delimiter: Field delimiter

        Yields:
            RegistryCase objects
        """
        filepath = Path(filepath)
        self.safe_logger.info(f"Parsing SEER file: {filepath.name}")

        count = 0
        errors = 0

        with open(filepath, encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                try:
                    case = self._parse_row(row, str(filepath.name))
                    if case:
                        yield case
                        count += 1
                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Log first 10 errors only
                        self.safe_logger.warning(f"Error parsing row: {e}")

        self.safe_logger.info_aggregate("Cases parsed", count)
        if errors:
            self.safe_logger.info_aggregate("Parse errors", errors)

    def _parse_row(self, row: dict[str, str], source: str) -> Optional[RegistryCase]:
        """Parse a single row into a RegistryCase.

        Args:
            row: Row dictionary
            source: Source file name

        Returns:
            RegistryCase or None if invalid
        """
        # Get patient ID and create hashed key
        patient_id = _find_field(row, SEER_FIELD_ALIASES["patient_id"])
        if not patient_id:
            return None

        patient_key = create_patient_key(
            patient_id,
            salt=self.privacy_config.salt,
            algorithm=self.privacy_config.hash_algorithm,
        )

        # Parse diagnosis date
        dx_date = _parse_date(
            _find_field(row, SEER_FIELD_ALIASES["dx_date"]),
            _find_field(row, SEER_FIELD_ALIASES["dx_year"]),
        )
        if not dx_date:
            return None

        # Parse other fields
        site = _find_field(row, SEER_FIELD_ALIASES["site"]) or ""
        histology = _find_field(row, SEER_FIELD_ALIASES["histology"]) or ""
        stage = _find_field(row, SEER_FIELD_ALIASES["stage"]) or ""
        grade = _find_field(row, SEER_FIELD_ALIASES["grade"]) or ""
        laterality = _find_field(row, SEER_FIELD_ALIASES["laterality"]) or ""

        # Parse age
        age_str = _find_field(row, SEER_FIELD_ALIASES["age"])
        age = int(age_str) if age_str and age_str.isdigit() else None

        # Demographics
        sex = _find_field(row, SEER_FIELD_ALIASES["sex"]) or ""
        race = _find_field(row, SEER_FIELD_ALIASES["race"]) or ""

        # Vital status and death
        vital_status = _parse_vital_status(_find_field(row, SEER_FIELD_ALIASES["vital_status"]))
        death_date = _parse_date(_find_field(row, SEER_FIELD_ALIASES["death_date"]))
        cause_of_death = _parse_cause_of_death(
            _find_field(row, SEER_FIELD_ALIASES["cause_of_death"]), site
        )

        # Survival
        surv_str = _find_field(row, SEER_FIELD_ALIASES["survival_months"])
        survival_months = int(surv_str) if surv_str and surv_str.isdigit() else None

        # Sequence number
        seq_str = _find_field(row, SEER_FIELD_ALIASES["sequence_number"])
        sequence_number = int(seq_str) if seq_str and seq_str.isdigit() else 0

        return RegistryCase(
            patient_key=patient_key,
            dx_date=dx_date,
            cancer_site=site,
            histology=histology,
            stage=stage,
            grade=grade,
            laterality=laterality,
            age_at_dx=age,
            sex=sex,
            race=race,
            vital_status=vital_status,
            death_date=death_date,
            cause_of_death=cause_of_death,
            survival_months=survival_months,
            sequence_number=sequence_number,
            raw_source=source,
        )

    def parse_directory(
        self,
        directory: Path,
        file_pattern: str = "*.csv",
    ) -> Generator[RegistryCase, None, None]:
        """Parse all matching files in a directory.

        Args:
            directory: Directory to scan
            file_pattern: Glob pattern for files

        Yields:
            RegistryCase objects
        """
        directory = Path(directory)
        self.safe_logger.info(f"Scanning directory: {directory}")

        for filepath in sorted(directory.glob(file_pattern)):
            yield from self.parse_file(filepath)


def filter_cases_by_site(
    cases: Generator[RegistryCase, None, None],
    site_patterns: list[str],
) -> Generator[RegistryCase, None, None]:
    """Filter registry cases by cancer site patterns.

    Args:
        cases: Generator of RegistryCase objects
        site_patterns: List of site patterns to match (case-insensitive)

    Yields:
        Matching RegistryCase objects
    """
    patterns_upper = [p.upper() for p in site_patterns]

    for case in cases:
        site_upper = case.cancer_site.upper()
        if any(p in site_upper for p in patterns_upper):
            yield case


def filter_cases_by_year(
    cases: Generator[RegistryCase, None, None],
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Generator[RegistryCase, None, None]:
    """Filter registry cases by diagnosis year.

    Args:
        cases: Generator of RegistryCase objects
        year_min: Minimum diagnosis year (inclusive)
        year_max: Maximum diagnosis year (inclusive)

    Yields:
        Matching RegistryCase objects
    """
    for case in cases:
        if case.dx_date is None:
            continue
        year = case.dx_date.year
        if year_min and year < year_min:
            continue
        if year_max and year > year_max:
            continue
        yield case

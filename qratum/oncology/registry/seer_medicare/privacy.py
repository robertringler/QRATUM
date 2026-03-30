"""
Privacy and Safe Logging Utilities for SEER-Medicare Pipeline

Provides utilities for:
- Safe logging that never exposes patient identifiers
- Data redaction and hashing
- Aggregation with minimum cell size suppression
- DUA compliance helpers

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Default minimum cell size for suppression (HIPAA Safe Harbor principle)
DEFAULT_MIN_CELL_SIZE = 11


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving operations.

    Attributes:
        min_cell_size: Minimum cell size for aggregation (default 11)
        salt: Salt for hashing operations (should be unique per DUA)
        hash_algorithm: Hash algorithm to use (default sha256)
        redact_patterns: Regex patterns for additional redaction
        log_aggregates_only: Only log aggregate statistics, never row-level
    """

    min_cell_size: int = DEFAULT_MIN_CELL_SIZE
    salt: str = ""
    hash_algorithm: str = "sha256"
    redact_patterns: list[str] = field(default_factory=list)
    log_aggregates_only: bool = True

    def __post_init__(self) -> None:
        """Compile regex patterns."""
        self._compiled_patterns = [re.compile(p) for p in self.redact_patterns]


class SafeLogger:
    """
    Privacy-preserving logger that never exposes patient identifiers.

    This logger:
    - Filters out potential patient identifiers from log messages
    - Only logs aggregated counts (never row-level data)
    - Enforces minimum cell size suppression
    - Redacts configured sensitive patterns

    Usage:
        config = PrivacyConfig(min_cell_size=11)
        logger = SafeLogger("seer_medicare", config)
        logger.info_aggregate("Patients in cohort", count=150)  # OK
        logger.info(f"Patient {patient_id}")  # Will be redacted
    """

    # Patterns that look like identifiers
    # Note: These patterns balance detection sensitivity with false positive rate.
    # Consider extending for specific data sources as needed.
    IDENTIFIER_PATTERNS = [
        r"\b[A-Z]{3}\d{9}\b",  # Medicare HIC-like
        r"\b\d{9}\b",  # SSN-like (9 digits)
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
        r"\bPATIENT_?\w*[:=]\s*\w+",  # Explicit patient_id patterns
        r"\bBENE_?\w*[:=]\s*\w+",  # Beneficiary ID patterns
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # Email addresses
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone numbers
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Date formats (potential DOB)
        r"\b\d{5}(-\d{4})?\b",  # ZIP codes (potential geographic identifier)
        r"\bMRN[:=]?\s*\d+\b",  # Medical record numbers
        r"\bNCH_?\w*[:=]\s*\w+",  # NCH claim identifiers
    ]

    def __init__(
        self,
        name: str,
        config: Optional[PrivacyConfig] = None,
        underlying_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize SafeLogger.

        Args:
            name: Logger name
            config: Privacy configuration
            underlying_logger: Optional underlying logger (creates one if None)
        """
        self.name = name
        self.config = config or PrivacyConfig()
        self._logger = underlying_logger or logging.getLogger(name)

        # Compile identifier patterns
        self._id_patterns = [re.compile(p, re.IGNORECASE) for p in self.IDENTIFIER_PATTERNS]

    def _redact_message(self, message: str) -> str:
        """Redact potential identifiers from message.

        Args:
            message: Log message

        Returns:
            Redacted message
        """
        result = message

        # Redact identifier patterns
        for pattern in self._id_patterns:
            result = pattern.sub("[REDACTED]", result)

        # Redact configured patterns
        for pattern in self.config._compiled_patterns:
            result = pattern.sub("[REDACTED]", result)

        return result

    def _format_aggregate(self, description: str, count: int) -> str:
        """Format an aggregate count with suppression.

        Args:
            description: Description of the count
            count: The count value

        Returns:
            Formatted message with suppression if needed
        """
        if count < self.config.min_cell_size:
            return f"{description}: <{self.config.min_cell_size} (suppressed)"
        return f"{description}: {count}"

    def debug(self, message: str) -> None:
        """Log debug message with redaction."""
        self._logger.debug(self._redact_message(message))

    def info(self, message: str) -> None:
        """Log info message with redaction."""
        self._logger.info(self._redact_message(message))

    def warning(self, message: str) -> None:
        """Log warning message with redaction."""
        self._logger.warning(self._redact_message(message))

    def error(self, message: str) -> None:
        """Log error message with redaction."""
        self._logger.error(self._redact_message(message))

    def info_aggregate(self, description: str, count: int) -> None:
        """Log an aggregate count with suppression.

        Args:
            description: Description of the metric
            count: Count value (suppressed if < min_cell_size)
        """
        self._logger.info(self._format_aggregate(description, count))

    def info_distribution(
        self,
        description: str,
        distribution: dict[str, int],
        sort_by_value: bool = True,
    ) -> None:
        """Log a distribution with suppression.

        Args:
            description: Description of the distribution
            distribution: Dictionary of category -> count
            sort_by_value: Sort by count (descending) if True
        """
        self._logger.info(f"{description}:")

        items = distribution.items()
        if sort_by_value:
            items = sorted(items, key=lambda x: x[1], reverse=True)

        for key, count in items:
            msg = self._format_aggregate(f"  {key}", count)
            self._logger.info(msg)

    def log_cohort_summary(
        self,
        total: int,
        included: int,
        excluded_reasons: dict[str, int],
    ) -> None:
        """Log a cohort summary with suppression.

        Args:
            total: Total cases considered
            included: Number included in cohort
            excluded_reasons: Exclusion reasons and counts
        """
        self._logger.info("=== Cohort Summary ===")
        self._logger.info(self._format_aggregate("Total cases", total))
        self._logger.info(self._format_aggregate("Included in cohort", included))

        if excluded_reasons:
            self._logger.info("Exclusions:")
            for reason, count in excluded_reasons.items():
                self._logger.info(self._format_aggregate(f"  {reason}", count))


def create_patient_key(
    original_id: str,
    salt: str = "",
    algorithm: str = "sha256",
) -> str:
    """Create a stable hashed patient key from original identifier.

    The hashed key is deterministic for the same input + salt,
    allowing consistent patient linking without exposing original IDs.

    Args:
        original_id: Original patient identifier
        salt: Salt for hashing (should be unique per DUA/study)
        algorithm: Hash algorithm (sha256, sha3_256, etc.)

    Returns:
        Hashed patient key (hex string, truncated to 16 chars)
    """
    combined = f"{salt}{original_id}".encode()

    if algorithm == "sha256":
        h = hashlib.sha256(combined)
    elif algorithm == "sha3_256":
        h = hashlib.sha3_256(combined)
    elif algorithm == "md5":
        # Not recommended but supported for legacy
        h = hashlib.md5(combined)
    else:
        h = hashlib.sha256(combined)

    return h.hexdigest()[:16]


def create_claim_id_hash(
    claim_id: str,
    salt: str = "",
) -> str:
    """Create a hashed claim identifier.

    Args:
        claim_id: Original claim ID
        salt: Salt for hashing

    Returns:
        Hashed claim ID (hex string, truncated to 12 chars)
    """
    combined = f"{salt}{claim_id}".encode()
    return hashlib.sha256(combined).hexdigest()[:12]


def suppress_small_counts(
    distribution: dict[str, int],
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
    suppress_value: str = "<N",
) -> dict[str, Any]:
    """Suppress small counts in a distribution.

    Args:
        distribution: Dictionary of category -> count
        min_cell_size: Minimum allowed count
        suppress_value: Value to use for suppressed cells

    Returns:
        Distribution with small counts suppressed
    """
    result = {}
    for key, count in distribution.items():
        if count < min_cell_size:
            result[key] = suppress_value.replace("N", str(min_cell_size))
        else:
            result[key] = count
    return result


def aggregate_age_to_bins(
    ages: list[int],
    bin_size: int = 5,
    min_age: int = 65,
    max_age: int = 100,
) -> dict[str, int]:
    """Aggregate ages into bins.

    Args:
        ages: List of ages
        bin_size: Size of each bin
        min_age: Minimum age for binning
        max_age: Maximum age for binning

    Returns:
        Dictionary of age_bin -> count
    """
    bins: dict[str, int] = {}

    for age in ages:
        if age is None:
            bin_label = "Unknown"
        elif age < min_age:
            bin_label = f"<{min_age}"
        elif age >= max_age:
            bin_label = f"{max_age}+"
        else:
            bin_start = (age // bin_size) * bin_size
            bin_end = bin_start + bin_size - 1
            bin_label = f"{bin_start}-{bin_end}"

        bins[bin_label] = bins.get(bin_label, 0) + 1

    return bins


def redact_value(value: Any, replacement: str = "[REDACTED]") -> str:
    """Redact a value for logging.

    Args:
        value: Value to redact
        replacement: Replacement string

    Returns:
        Replacement string
    """
    return replacement


class DUAComplianceChecker:
    """
    Helper class for DUA compliance checks.

    Provides methods to validate that outputs comply with
    data use agreement requirements.
    """

    def __init__(self, config: Optional[PrivacyConfig] = None) -> None:
        """Initialize checker.

        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.issues: list[str] = []

    def check_cell_sizes(self, distribution: dict[str, int]) -> bool:
        """Check that all cell sizes meet minimum threshold.

        Args:
            distribution: Distribution to check

        Returns:
            True if all cells pass, False otherwise
        """
        passed = True
        for key, count in distribution.items():
            if 0 < count < self.config.min_cell_size:
                self.issues.append(f"Cell '{key}' has count {count} < {self.config.min_cell_size}")
                passed = False
        return passed

    def check_no_identifiers(self, text: str) -> bool:
        """Check that text contains no identifier patterns.

        Args:
            text: Text to check

        Returns:
            True if no identifiers found, False otherwise
        """
        for pattern in SafeLogger.IDENTIFIER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                self.issues.append(f"Potential identifier found matching pattern: {pattern}")
                return False
        return True

    def generate_checklist(self) -> dict[str, Any]:
        """Generate a DUA compliance checklist.

        Returns:
            Checklist dictionary
        """
        return {
            "checklist": [
                {
                    "item": "No patient-level identifiers in output",
                    "status": "REQUIRES_VERIFICATION",
                    "notes": "Verify all patient keys are hashed, no original IDs exposed",
                },
                {
                    "item": f"All cell counts >= {self.config.min_cell_size}",
                    "status": "REQUIRES_VERIFICATION",
                    "notes": "Cells with smaller counts must be suppressed",
                },
                {
                    "item": "No network upload of data",
                    "status": "AUTOMATED_CHECK",
                    "notes": "Pipeline runs locally only",
                },
                {
                    "item": "Outputs stored in designated directory",
                    "status": "REQUIRES_VERIFICATION",
                    "notes": "All outputs in artifacts/seer_medicare_run_*/",
                },
                {
                    "item": "Data manifest includes file hashes",
                    "status": "REQUIRES_VERIFICATION",
                    "notes": "SHA256 hashes computed for all input files",
                },
                {
                    "item": "Reproducibility artifacts generated",
                    "status": "REQUIRES_VERIFICATION",
                    "notes": "run_config.json, environment.txt, git_commit.txt",
                },
            ],
            "issues_found": self.issues,
            "min_cell_size": self.config.min_cell_size,
        }

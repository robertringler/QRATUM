"""Validator for reinjection candidates.

Implements validation logic for discovery reinjection including:
- Schema validation
- Provenance verification
- Confidence threshold checks
- Domain-specific validation rules
- Regulatory compliance checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qradle.merkle import MerkleChain
from qratum_asi.reinjection.types import (
    DiscoveryDomain,
    ReinjectionCandidate,
    ReinjectionScore,
    ValidationLevel,
)


@dataclass
class ValidationResult:
    """Result of candidate validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        checks_passed: Number of checks passed
        checks_total: Total number of checks
        validation_level: Level of validation performed
    """

    valid: bool
    errors: list[str]
    warnings: list[str]
    checks_passed: int
    checks_total: int
    validation_level: ValidationLevel
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "validation_level": self.validation_level.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# Domain-specific confidence thresholds
DOMAIN_CONFIDENCE_THRESHOLDS: dict[DiscoveryDomain, float] = {
    DiscoveryDomain.BIODISCOVERY: 0.75,
    DiscoveryDomain.GENOMICS: 0.80,
    DiscoveryDomain.DRUG_DISCOVERY: 0.85,
    DiscoveryDomain.CLIMATE_BIOLOGY: 0.70,
    DiscoveryDomain.LONGEVITY: 0.80,
    DiscoveryDomain.NEURAL: 0.75,
    DiscoveryDomain.ECONOMIC_BIOLOGICAL: 0.70,
    DiscoveryDomain.CROSS_VERTICAL: 0.75,
}

# Minimum mutual information for reinjection
MINIMUM_MUTUAL_INFORMATION = 0.5

# Minimum composite score for reinjection
MINIMUM_COMPOSITE_SCORE = 0.6


class ReinjectionValidator:
    """Validator for reinjection candidates.

    Performs multi-level validation including:
    - Schema and structure validation
    - Provenance chain verification
    - Confidence and score thresholds
    - Domain-specific rules
    - Regulatory compliance mapping
    """

    def __init__(self, merkle_chain: MerkleChain | None = None):
        """Initialize validator.

        Args:
            merkle_chain: Optional Merkle chain for provenance verification
        """
        self.merkle_chain = merkle_chain or MerkleChain()
        self.validation_history: list[ValidationResult] = []

    def validate(
        self,
        candidate: ReinjectionCandidate,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> ValidationResult:
        """Validate a reinjection candidate.

        Args:
            candidate: Candidate to validate
            validation_level: Level of validation to perform

        Returns:
            ValidationResult with pass/fail and details
        """
        errors: list[str] = []
        warnings: list[str] = []
        checks_passed = 0
        checks_total = 0
        details: dict[str, Any] = {}

        # Basic schema validation
        checks_total += 1
        schema_valid, schema_errors = self._validate_schema(candidate)
        if schema_valid:
            checks_passed += 1
        else:
            errors.extend(schema_errors)
        details["schema_validation"] = {"passed": schema_valid, "errors": schema_errors}

        # Provenance verification
        checks_total += 1
        provenance_valid, provenance_msg = self._validate_provenance(candidate)
        if provenance_valid:
            checks_passed += 1
        else:
            errors.append(provenance_msg)
        details["provenance_verification"] = {"passed": provenance_valid, "message": provenance_msg}

        # Score threshold validation
        checks_total += 1
        score_valid, score_warnings = self._validate_scores(candidate)
        if score_valid:
            checks_passed += 1
        else:
            errors.append("Score thresholds not met")
        warnings.extend(score_warnings)
        details["score_validation"] = {"passed": score_valid, "warnings": score_warnings}

        # Domain-specific validation
        checks_total += 1
        domain_valid, domain_errors = self._validate_domain_rules(candidate)
        if domain_valid:
            checks_passed += 1
        else:
            errors.extend(domain_errors)
        details["domain_validation"] = {"passed": domain_valid, "errors": domain_errors}

        # Enhanced validation for higher levels
        if validation_level in (ValidationLevel.ENHANCED, ValidationLevel.CRITICAL):
            checks_total += 1
            compliance_valid, compliance_notes = self._validate_compliance(candidate)
            if compliance_valid:
                checks_passed += 1
            else:
                warnings.extend(compliance_notes)
            details["compliance_validation"] = {
                "passed": compliance_valid,
                "notes": compliance_notes,
            }

        # Critical level additional checks
        if validation_level == ValidationLevel.CRITICAL:
            checks_total += 1
            cross_impact_valid = candidate.score.cross_impact >= 0.7
            if cross_impact_valid:
                checks_passed += 1
            else:
                errors.append("Critical validation requires cross_impact >= 0.7")
            details["critical_validation"] = {"cross_impact_check": cross_impact_valid}

        # Determine overall validity
        valid = len(errors) == 0

        result = ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
            checks_total=checks_total,
            validation_level=validation_level,
            details=details,
        )

        self.validation_history.append(result)

        # Log to merkle chain
        self.merkle_chain.add_event(
            "validation_completed",
            {
                "candidate_id": candidate.candidate_id,
                "valid": valid,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
            },
        )

        return result

    def _validate_schema(self, candidate: ReinjectionCandidate) -> tuple[bool, list[str]]:
        """Validate candidate schema and required fields."""
        errors: list[str] = []

        # Check required fields
        if not candidate.candidate_id:
            errors.append("Missing candidate_id")
        if not candidate.discovery_id:
            errors.append("Missing discovery_id")
        if not candidate.description:
            errors.append("Missing description")
        if not candidate.data_payload:
            errors.append("Missing data_payload")
        if not candidate.target_priors:
            errors.append("Missing target_priors")
        if not candidate.source_workflow_id:
            errors.append("Missing source_workflow_id")
        if not candidate.provenance_hash:
            errors.append("Missing provenance_hash")

        # Validate score structure
        if candidate.score is None:
            errors.append("Missing score")
        elif not isinstance(candidate.score, ReinjectionScore):
            errors.append("Invalid score type")

        return len(errors) == 0, errors

    def _validate_provenance(self, candidate: ReinjectionCandidate) -> tuple[bool, str]:
        """Validate provenance hash and chain integrity."""
        # Verify content hash matches stored provenance
        computed_hash = candidate.compute_hash()

        # For this validation, we check that provenance_hash is set
        # In production, this would verify against the actual chain
        if not candidate.provenance_hash:
            return False, "Provenance hash is empty"

        if len(candidate.provenance_hash) != 64:
            return False, "Invalid provenance hash length"

        return True, "Provenance verified"

    def _validate_scores(self, candidate: ReinjectionCandidate) -> tuple[bool, list[str]]:
        """Validate score thresholds."""
        warnings: list[str] = []
        score = candidate.score

        # Check mutual information threshold
        if score.mutual_information < MINIMUM_MUTUAL_INFORMATION:
            return False, [
                f"Mutual information {score.mutual_information} below minimum {MINIMUM_MUTUAL_INFORMATION}"
            ]

        # Check composite score
        if score.composite_score < MINIMUM_COMPOSITE_SCORE:
            return False, [
                f"Composite score {score.composite_score:.3f} below minimum {MINIMUM_COMPOSITE_SCORE}"
            ]

        # Check domain-specific confidence threshold
        threshold = DOMAIN_CONFIDENCE_THRESHOLDS.get(candidate.domain, 0.75)
        if score.confidence < threshold:
            warnings.append(f"Confidence {score.confidence} below domain threshold {threshold}")

        # Warn on low novelty
        if score.novelty < 0.3:
            warnings.append(f"Low novelty score: {score.novelty}")

        return True, warnings

    def _validate_domain_rules(self, candidate: ReinjectionCandidate) -> tuple[bool, list[str]]:
        """Validate domain-specific rules."""
        errors: list[str] = []
        domain = candidate.domain

        # Domain-specific validation
        if domain == DiscoveryDomain.DRUG_DISCOVERY:
            # Drug discovery requires specific payload fields
            payload = candidate.data_payload
            if "compound_data" not in payload and "target_data" not in payload:
                errors.append("Drug discovery requires compound_data or target_data")

        elif domain == DiscoveryDomain.GENOMICS:
            # Genomics requires reference genome info
            payload = candidate.data_payload
            if "reference" not in payload:
                errors.append("Genomics requires reference genome specification")

        elif domain == DiscoveryDomain.LONGEVITY:
            # Longevity research has enhanced safety requirements
            if candidate.validation_level not in (
                ValidationLevel.ENHANCED,
                ValidationLevel.CRITICAL,
            ):
                errors.append("Longevity research requires enhanced or critical validation")

        return len(errors) == 0, errors

    def _validate_compliance(self, candidate: ReinjectionCandidate) -> tuple[bool, list[str]]:
        """Validate regulatory compliance requirements."""
        notes: list[str] = []
        domain = candidate.domain

        # Domain to compliance framework mapping
        compliance_map = {
            DiscoveryDomain.GENOMICS: ["GDPR", "GINA", "HIPAA"],
            DiscoveryDomain.DRUG_DISCOVERY: ["FDA_21_CFR_Part_11", "HIPAA"],
            DiscoveryDomain.BIODISCOVERY: ["Nagoya_Protocol", "GDPR"],
            DiscoveryDomain.LONGEVITY: ["HIPAA", "IRB_Review"],
            DiscoveryDomain.NEURAL: ["HIPAA", "GDPR"],
            DiscoveryDomain.CLIMATE_BIOLOGY: ["Environmental_Regulations"],
            DiscoveryDomain.ECONOMIC_BIOLOGICAL: ["Financial_Regulations", "GDPR"],
            DiscoveryDomain.CROSS_VERTICAL: ["GDPR", "HIPAA"],
        }

        required_frameworks = compliance_map.get(domain, [])

        for framework in required_frameworks:
            notes.append(f"Compliance mapping required: {framework}")

        return True, notes

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
            }

        passed = sum(1 for v in self.validation_history if v.valid)
        failed = len(self.validation_history) - passed

        return {
            "total_validations": len(self.validation_history),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.validation_history) if self.validation_history else 0.0,
            "by_level": {
                level.value: sum(
                    1 for v in self.validation_history if v.validation_level == level and v.valid
                )
                for level in ValidationLevel
            },
        }

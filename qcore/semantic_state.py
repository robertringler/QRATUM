"""Canonical semantic problem definition with validation.

This module provides the SemanticState dataclass for representing quantum computing
problems in a domain-agnostic way, along with domain-specific validators.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticState:
    """Canonical representation of a semantic problem state.

    Attributes:
        variables: Dictionary of problem variables
        constraints: Dictionary of problem constraints
        objective: Problem objective function or goal
        domain: Problem domain (e.g., 'chemistry', 'optimization', 'finance')
        metadata: Additional domain-specific metadata
    """

    variables: dict[str, Any]
    constraints: dict[str, Any]
    objective: Any
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the semantic state after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate semantic state constraints.

        Raises:
            ValueError: If validation fails
        """
        if not self.variables:
            raise ValueError("variables must be non-empty")
        if self.objective is None:
            raise ValueError("objective must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")

        # Apply domain-specific validation
        validator = DomainValidator.get(self.domain)
        if validator is not None:
            validator.validate(self)

    def serialize(self) -> dict[str, Any]:
        """Return deterministic dict representation.

        Returns:
            Dictionary with all state fields in sorted order
        """
        return {
            "constraints": self.constraints,
            "domain": self.domain,
            "metadata": self.metadata,
            "objective": self.objective,
            "variables": self.variables,
        }

    def to_json(self) -> str:
        """Return JSON with sorted keys for deterministic serialization.

        Returns:
            JSON string representation with sorted keys
        """
        return json.dumps(self.serialize(), sort_keys=True, default=str)

    def compute_hash(self) -> str:
        """Return SHA-256 hash of canonical JSON serialization.

        Returns:
            Hexadecimal SHA-256 hash string
        """
        json_str = self.to_json()
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class DomainValidator:
    """Base class for domain-specific semantic state validators.

    Validators can be registered for specific domains to enforce
    domain-specific constraints.
    """

    _validators: dict[str, DomainValidator] = {}

    @classmethod
    def register(cls, domain: str, validator: DomainValidator) -> None:
        """Register a validator for a specific domain.

        Args:
            domain: Domain name (e.g., 'chemistry', 'optimization')
            validator: Validator instance for the domain
        """
        cls._validators[domain] = validator

    @classmethod
    def get(cls, domain: str) -> DomainValidator | None:
        """Get validator for a domain.

        Args:
            domain: Domain name

        Returns:
            Validator instance or None if not registered
        """
        return cls._validators.get(domain)

    def validate(self, state: SemanticState) -> None:
        """Validate a semantic state.

        Override this method in subclasses to implement domain-specific validation.

        Args:
            state: Semantic state to validate

        Raises:
            ValueError: If validation fails
        """
        pass


class ChemistryValidator(DomainValidator):
    """Validator for chemistry domain problems.

    Requires 'molecule' key in metadata.
    """

    def validate(self, state: SemanticState) -> None:
        """Validate chemistry domain semantic state.

        Args:
            state: Semantic state to validate

        Raises:
            ValueError: If molecule not specified in metadata
        """
        if "molecule" not in state.metadata:
            raise ValueError("Chemistry domain requires 'molecule' in metadata")


class OptimizationValidator(DomainValidator):
    """Validator for optimization domain problems.

    Requires non-empty constraints.
    """

    def validate(self, state: SemanticState) -> None:
        """Validate optimization domain semantic state.

        Args:
            state: Semantic state to validate

        Raises:
            ValueError: If constraints are empty
        """
        if not state.constraints:
            raise ValueError("Optimization domain requires non-empty constraints")


class FinanceValidator(DomainValidator):
    """Validator for finance domain problems.

    Requires 'risk_tolerance' key in metadata.
    """

    def validate(self, state: SemanticState) -> None:
        """Validate finance domain semantic state.

        Args:
            state: Semantic state to validate

        Raises:
            ValueError: If risk_tolerance not specified in metadata
        """
        if "risk_tolerance" not in state.metadata:
            raise ValueError("Finance domain requires 'risk_tolerance' in metadata")


# Register default validators
DomainValidator.register("chemistry", ChemistryValidator())
DomainValidator.register("optimization", OptimizationValidator())
DomainValidator.register("finance", FinanceValidator())

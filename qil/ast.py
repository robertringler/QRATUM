"""QIL Abstract Syntax Tree Models.

This module defines the AST node types for QIL intents,
representing the parsed structure of intent declarations.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Objective:
    """Represents an OBJECTIVE statement in QIL.

    Attributes:
        name: Objective identifier
    """

    name: str


@dataclass(frozen=True)
class Constraint:
    """Represents a CONSTRAINT statement in QIL.

    Attributes:
        name: Constraint identifier (e.g., "GPU_VRAM")
        operator: Comparison operator (>=, <=, ==, >, <, !=)
        value: Constraint value (number, string, or identifier)
    """

    name: str
    operator: str
    value: Any


@dataclass(frozen=True)
class Capability:
    """Represents a CAPABILITY statement in QIL.

    Attributes:
        name: Capability identifier
    """

    name: str


@dataclass(frozen=True)
class TimeSpec:
    """Represents a TIME specification in QIL.

    Attributes:
        key: Time key (deadline, budget, window)
        value: Time value (number)
        unit: Time unit (s, ms, h, m)
    """

    key: str
    value: float
    unit: str

    def to_seconds(self) -> float:
        """Convert time specification to seconds.

        Returns:
            Time value in seconds
        """
        unit_multipliers = {
            "s": 1.0,
            "ms": 0.001,
            "m": 60.0,
            "h": 3600.0,
        }
        return self.value * unit_multipliers.get(self.unit, 1.0)


@dataclass(frozen=True)
class Authority:
    """Represents an AUTHORITY statement in QIL.

    Attributes:
        key: Authority key (user, group, role)
        value: Authority identifier
    """

    key: str
    value: str


@dataclass(frozen=True)
class Trust:
    """Represents a TRUST statement in QIL.

    Attributes:
        level: Trust level (verified, trusted, untrusted, sandbox)
    """

    level: str


@dataclass(frozen=True)
class HardwareSpec:
    """Represents a HARDWARE specification in QIL.

    Attributes:
        only_clusters: List of clusters to exclusively use (ONLY clause)
        not_clusters: List of clusters to exclude (NOT clause)
    """

    only_clusters: list[str] = field(default_factory=list)
    not_clusters: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate hardware specification after initialization."""
        # Check for conflicting specifications
        only_set = set(self.only_clusters)
        not_set = set(self.not_clusters)
        overlap = only_set & not_set
        if overlap:
            raise ValueError(
                f"Hardware specification conflict: clusters {overlap} "
                "appear in both ONLY and NOT clauses"
            )


@dataclass(frozen=True)
class Intent:
    """Represents a complete QIL INTENT declaration.

    Attributes:
        name: Intent identifier
        objective: Objective statement (required)
        constraints: List of constraint statements
        capabilities: List of capability statements
        time_specs: List of time specifications
        authorities: List of authority statements
        trust: Optional trust specification
        hardware: Optional hardware specification
        metadata: Additional metadata
    """

    name: str
    objective: Objective
    constraints: list[Constraint] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    time_specs: list[TimeSpec] = field(default_factory=list)
    authorities: list[Authority] = field(default_factory=list)
    trust: Trust | None = None
    hardware: HardwareSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate intent after initialization."""
        if not self.name:
            raise ValueError("Intent name cannot be empty")
        if not self.objective:
            raise ValueError("Intent must have an objective")

    def serialize(self) -> dict[str, Any]:
        """Serialize intent to dictionary representation.

        Returns:
            Dictionary representation of intent
        """
        result: dict[str, Any] = {
            "name": self.name,
            "objective": self.objective.name,
            "constraints": [
                {
                    "name": c.name,
                    "operator": c.operator,
                    "value": c.value,
                }
                for c in self.constraints
            ],
            "capabilities": [c.name for c in self.capabilities],
            "time_specs": [
                {
                    "key": t.key,
                    "value": t.value,
                    "unit": t.unit,
                }
                for t in self.time_specs
            ],
            "authorities": [
                {
                    "key": a.key,
                    "value": a.value,
                }
                for a in self.authorities
            ],
        }

        if self.trust:
            result["trust"] = self.trust.level

        if self.hardware:
            result["hardware"] = {
                "only_clusters": self.hardware.only_clusters,
                "not_clusters": self.hardware.not_clusters,
            }

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def get_deadline_seconds(self) -> float | None:
        """Get deadline time specification in seconds.

        Returns:
            Deadline in seconds, or None if not specified
        """
        for time_spec in self.time_specs:
            if time_spec.key == "deadline":
                return time_spec.to_seconds()
        return None

    def get_authority_user(self) -> str | None:
        """Get user authority if specified.

        Returns:
            User identifier, or None if not specified
        """
        for authority in self.authorities:
            if authority.key == "user":
                return authority.value
        return None

    def requires_cluster(self, cluster_type: str) -> bool:
        """Check if intent requires a specific cluster type.

        Args:
            cluster_type: Cluster type to check

        Returns:
            True if cluster is required, False otherwise
        """
        if not self.hardware:
            return False

        # If ONLY clause exists, cluster must be in it
        if self.hardware.only_clusters:
            return cluster_type in self.hardware.only_clusters

        # If NOT clause exists, cluster must not be in it
        if self.hardware.not_clusters:
            return cluster_type not in self.hardware.not_clusters

        return False

    def is_cluster_excluded(self, cluster_type: str) -> bool:
        """Check if a cluster type is explicitly excluded.

        Args:
            cluster_type: Cluster type to check

        Returns:
            True if cluster is excluded, False otherwise
        """
        if not self.hardware:
            return False

        # Excluded if in NOT clause
        if cluster_type in self.hardware.not_clusters:
            return True

        # Excluded if ONLY clause exists and cluster not in it
        return bool(self.hardware.only_clusters and cluster_type not in self.hardware.only_clusters)

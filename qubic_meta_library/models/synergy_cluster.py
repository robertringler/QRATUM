"""Synergy cluster data model for Qubic Meta Library."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynergyCluster:
    """

    Represents a synergy cluster connecting multiple domains.

    Attributes:
        id: Cluster identifier (A-DK, 110 total)
        name: Cluster name
        domains: List of domain IDs in cluster
        prompts: List of prompt IDs in cluster
        application: Primary application area
        execution_mode: Execution mode (sequential/parallel/hybrid)
        revenue_projection: Projected revenue (2026-2030)
        cluster_type: Type of cluster (two-domain/multi-domain/full-stack)
    """

    id: str
    name: str
    domains: list[str]
    prompts: list[int] = field(default_factory=list)
    application: str = ""
    execution_mode: str = "parallel"
    revenue_projection: dict[int, float] = field(default_factory=dict)
    cluster_type: str = "two-domain"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate synergy cluster attributes."""

        if not self.domains:
            raise ValueError("Synergy cluster must contain at least one domain")
        if self.execution_mode not in ["sequential", "parallel", "hybrid"]:
            raise ValueError(
                f"Execution mode must be sequential, parallel, or hybrid, got {self.execution_mode}"
            )

        # Auto-detect cluster type based on domain count
        if len(self.domains) == 2:
            self.cluster_type = "two-domain"
        elif len(self.domains) == 20:
            self.cluster_type = "full-stack"
        elif len(self.domains) > 2:
            self.cluster_type = "multi-domain"

        # Validate the final cluster type
        if self.cluster_type not in ["two-domain", "multi-domain", "full-stack"]:
            raise ValueError(
                f"Cluster type must be two-domain, multi-domain, or full-stack, got {self.cluster_type}"
            )

    def get_total_revenue_projection(self) -> float:
        """Calculate total revenue projection across all years."""

        return sum(self.revenue_projection.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert synergy cluster to dictionary representation."""

        return {
            "id": self.id,
            "name": self.name,
            "domains": self.domains,
            "prompts": self.prompts,
            "application": self.application,
            "execution_mode": self.execution_mode,
            "revenue_projection": self.revenue_projection,
            "cluster_type": self.cluster_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynergyCluster":
        """Create SynergyCluster instance from dictionary."""

        return cls(
            id=data["id"],
            name=data["name"],
            domains=data["domains"],
            prompts=data.get("prompts", []),
            application=data.get("application", ""),
            execution_mode=data.get("execution_mode", "parallel"),
            revenue_projection=data.get("revenue_projection", {}),
            cluster_type=data.get("cluster_type", "two-domain"),
            metadata=data.get("metadata", {}),
        )

"""Prompt data model for Qubic Meta Library."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prompt:
    """
    Represents a prompt in the Qubic Meta Library.

    Attributes:
        id: Unique identifier (1-10000)
        category: Category classification
        description: Detailed prompt description
        domain: Domain identifier (D1-D20 or Integration)
        patentability_score: Patent potential score (0.0-1.0)
        commercial_potential: Commercial viability score (0.0-1.0)
        keystone_nodes: List of keystone technology nodes
        synergy_connections: List of connected domain IDs
        execution_layers: Execution layer assignments (QuASIM/QStack/QNimbus)
        phase_deployment: Deployment phase (1-4)
        output_type: Expected output type (simulation/model/analysis/etc)
    """

    id: int
    category: str
    description: str
    domain: str
    patentability_score: float
    commercial_potential: float
    keystone_nodes: list[str] = field(default_factory=list)
    synergy_connections: list[str] = field(default_factory=list)
    execution_layers: list[str] = field(default_factory=list)
    phase_deployment: int = 1
    output_type: str = "simulation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate prompt attributes."""
        if not 1 <= self.id <= 10000:
            raise ValueError(f"Prompt ID must be between 1 and 10000, got {self.id}")
        if not 0.0 <= self.patentability_score <= 1.0:
            raise ValueError(
                f"Patentability score must be between 0.0 and 1.0, got {self.patentability_score}"
            )
        if not 0.0 <= self.commercial_potential <= 1.0:
            raise ValueError(
                f"Commercial potential must be between 0.0 and 1.0, got {self.commercial_potential}"
            )
        if not 1 <= self.phase_deployment <= 4:
            raise ValueError(
                f"Phase deployment must be between 1 and 4, got {self.phase_deployment}"
            )

    def is_high_value(self, threshold: float = 0.8) -> bool:
        """
        Check if prompt is high-value based on patentability and commercial scores.

        Args:
            threshold: Minimum score threshold (default: 0.8)

        Returns:
            True if both scores exceed threshold
        """
        return self.patentability_score >= threshold and self.commercial_potential >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert prompt to dictionary representation."""
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "domain": self.domain,
            "patentability_score": self.patentability_score,
            "commercial_potential": self.commercial_potential,
            "keystone_nodes": self.keystone_nodes,
            "synergy_connections": self.synergy_connections,
            "execution_layers": self.execution_layers,
            "phase_deployment": self.phase_deployment,
            "output_type": self.output_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        """Create Prompt instance from dictionary."""
        return cls(
            id=data["id"],
            category=data["category"],
            description=data["description"],
            domain=data["domain"],
            patentability_score=data["patentability_score"],
            commercial_potential=data["commercial_potential"],
            keystone_nodes=data.get("keystone_nodes", []),
            synergy_connections=data.get("synergy_connections", []),
            execution_layers=data.get("execution_layers", []),
            phase_deployment=data.get("phase_deployment", 1),
            output_type=data.get("output_type", "simulation"),
            metadata=data.get("metadata", {}),
        )

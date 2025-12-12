"""Domain data model for Qubic Meta Library."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Domain:
    """
    Represents a domain in the Qubic Meta Library.

    Attributes:
        id: Domain identifier (D1-D20)
        name: Domain name
        tier: Domain tier (1-5)
        id_range: Tuple of (start_id, end_id) for prompt IDs
        primary_platform: Primary execution platform (QuASIM/QStack/QNimbus)
        commercial_sector: Target commercial sector
        keystones: List of keystone technology nodes
        description: Detailed domain description
    """

    id: str
    name: str
    tier: int
    id_range: tuple[int, int]
    primary_platform: str
    commercial_sector: str
    keystones: list[str] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate domain attributes."""
        if not self.id.startswith("D"):
            raise ValueError(f"Domain ID must start with 'D', got {self.id}")
        if not 1 <= self.tier <= 5:
            raise ValueError(f"Domain tier must be between 1 and 5, got {self.tier}")
        if self.id_range[0] >= self.id_range[1]:
            raise ValueError(
                f"Invalid ID range: start {self.id_range[0]} >= end {self.id_range[1]}"
            )
        if self.primary_platform not in ["QuASIM", "QStack", "QNimbus"]:
            raise ValueError(
                f"Primary platform must be QuASIM, QStack, or QNimbus, got {self.primary_platform}"
            )

    def contains_id(self, prompt_id: int) -> bool:
        """
        Check if prompt ID belongs to this domain.

        Args:
            prompt_id: Prompt ID to check

        Returns:
            True if prompt ID is within domain's range
        """
        return self.id_range[0] <= prompt_id <= self.id_range[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert domain to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "id_range": list(self.id_range),
            "primary_platform": self.primary_platform,
            "commercial_sector": self.commercial_sector,
            "keystones": self.keystones,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Domain":
        """Create Domain instance from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            tier=data["tier"],
            id_range=tuple(data["id_range"]),
            primary_platform=data["primary_platform"],
            commercial_sector=data["commercial_sector"],
            keystones=data.get("keystones", []),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

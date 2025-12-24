"""Intent Contract - Authorized Intent with Resolved Capabilities.

This module implements the IntentContract, which represents an authorized
intent with resolved capabilities ready for execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.base import (BaseContract, generate_contract_id,
                            get_current_timestamp)
from qil.ast import Intent


@dataclass(frozen=True)
class IntentContract(BaseContract):
    """Immutable contract representing an authorized intent.

    Attributes:
        intent_name: Name of the original intent
        intent_hash: SHA-256 hash of the original intent
        objective: Intent objective
        constraints: List of constraints
        capabilities: List of resolved capabilities
        time_specs: List of time specifications
        authorities: List of authorities
        trust_level: Trust level
        hardware_spec: Hardware requirements
        authorization_proof: Proof of authorization
    """

    intent_name: str = ""
    intent_hash: str = ""
    objective: str = ""
    constraints: list[dict[str, Any]] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    time_specs: list[dict[str, Any]] = field(default_factory=list)
    authorities: list[dict[str, Any]] = field(default_factory=list)
    trust_level: str = ""
    hardware_spec: dict[str, list[str]] = field(default_factory=dict)
    authorization_proof: str = ""

    def __post_init__(self) -> None:
        """Validate intent contract after initialization."""
        super().__post_init__()
        if not self.intent_name:
            raise ValueError("intent_name cannot be empty")
        if not self.intent_hash:
            raise ValueError("intent_hash cannot be empty")
        if not self.objective:
            raise ValueError("objective cannot be empty")
        if not self.authorization_proof:
            raise ValueError("authorization_proof cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize intent contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "intent_name": self.intent_name,
                "intent_hash": self.intent_hash,
                "objective": self.objective,
                "constraints": self.constraints,
                "capabilities": self.capabilities,
                "time_specs": self.time_specs,
                "authorities": self.authorities,
                "trust_level": self.trust_level,
                "hardware_spec": self.hardware_spec,
                "authorization_proof": self.authorization_proof,
            }
        )
        return base

    def get_deadline_seconds(self) -> float | None:
        """Get deadline in seconds if specified.

        Returns:
            Deadline in seconds, or None if not specified
        """
        for time_spec in self.time_specs:
            if time_spec.get("key") == "deadline":
                value = time_spec.get("value", 0)
                unit = time_spec.get("unit", "s")
                multipliers = {"s": 1.0, "ms": 0.001, "m": 60.0, "h": 3600.0}
                return value * multipliers.get(unit, 1.0)
        return None

    def requires_cluster(self, cluster_type: str) -> bool:
        """Check if contract requires specific cluster type.

        Args:
            cluster_type: Cluster type to check

        Returns:
            True if cluster is required, False otherwise
        """
        if not self.hardware_spec:
            return False

        only_clusters = self.hardware_spec.get("only_clusters", [])
        not_clusters = self.hardware_spec.get("not_clusters", [])

        # If ONLY clause exists, cluster must be in it
        if only_clusters:
            return cluster_type in only_clusters

        # If NOT clause exists, cluster must not be in it
        if not_clusters:
            return cluster_type not in not_clusters

        return False


def create_intent_contract(
    intent: Intent,
    authorization_proof: str,
    resolved_capabilities: list[str] | None = None,
) -> IntentContract:
    """Create an IntentContract from a validated Intent.

    Args:
        intent: Validated Intent object
        authorization_proof: Proof of authorization
        resolved_capabilities: Optional resolved capability list

    Returns:
        Immutable IntentContract
    """
    from qil.serializer import compute_hash

    intent_hash = compute_hash(intent)
    intent_serialized = intent.serialize()

    # Build contract content
    content = {
        "intent_name": intent.name,
        "intent_hash": intent_hash,
        "objective": intent_serialized["objective"],
        "constraints": intent_serialized["constraints"],
        "capabilities": resolved_capabilities or intent_serialized["capabilities"],
        "time_specs": intent_serialized["time_specs"],
        "authorities": intent_serialized["authorities"],
        "trust_level": intent_serialized.get("trust", "untrusted"),
        "hardware_spec": intent_serialized.get("hardware", {}),
        "authorization_proof": authorization_proof,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    # Generate contract ID
    contract_id = generate_contract_id("IntentContract", content)

    return IntentContract(
        contract_id=contract_id,
        contract_type="IntentContract",
        created_at=content["created_at"],
        version=content["version"],
        intent_name=content["intent_name"],
        intent_hash=content["intent_hash"],
        objective=content["objective"],
        constraints=content["constraints"],
        capabilities=content["capabilities"],
        time_specs=content["time_specs"],
        authorities=content["authorities"],
        trust_level=content["trust_level"],
        hardware_spec=content["hardware_spec"],
        authorization_proof=content["authorization_proof"],
    )

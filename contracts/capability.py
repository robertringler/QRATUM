"""Capability Contract - Hardware Binding with Cluster Topology.

This module implements the CapabilityContract and ClusterTopology,
representing hardware bindings for intent execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.base import (BaseContract, generate_contract_id,
                            get_current_timestamp)


@dataclass(frozen=True)
class ClusterTopology:
    """Immutable cluster topology specification.

    Attributes:
        cluster_type: Type of cluster (GB200, MI300X, etc.)
        node_count: Number of nodes in cluster
        accelerators_per_node: Number of accelerators per node
        memory_per_node_gb: Memory per node in GB
        interconnect: Interconnect type (NVLink, Infinity Fabric, etc.)
        metadata: Additional topology metadata
    """

    cluster_type: str
    node_count: int
    accelerators_per_node: int
    memory_per_node_gb: float
    interconnect: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cluster topology after initialization."""
        if not self.cluster_type:
            raise ValueError("cluster_type cannot be empty")
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        if self.accelerators_per_node <= 0:
            raise ValueError("accelerators_per_node must be positive")
        if self.memory_per_node_gb <= 0:
            raise ValueError("memory_per_node_gb must be positive")

    def total_accelerators(self) -> int:
        """Calculate total number of accelerators.

        Returns:
            Total accelerator count
        """
        return self.node_count * self.accelerators_per_node

    def total_memory_gb(self) -> float:
        """Calculate total cluster memory.

        Returns:
            Total memory in GB
        """
        return self.node_count * self.memory_per_node_gb

    def serialize(self) -> dict[str, Any]:
        """Serialize topology to dictionary."""
        return {
            "cluster_type": self.cluster_type,
            "node_count": self.node_count,
            "accelerators_per_node": self.accelerators_per_node,
            "memory_per_node_gb": self.memory_per_node_gb,
            "interconnect": self.interconnect,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CapabilityContract(BaseContract):
    """Immutable contract representing hardware capability binding.

    Attributes:
        intent_contract_id: Reference to IntentContract
        cluster_topology: Cluster topology specification
        allocated_resources: Allocated resource details
        resource_reservation_id: Reservation ID for resources
        capability_proof: Proof of capability binding
    """

    intent_contract_id: str = ""
    cluster_topology: dict[str, Any] = field(default_factory=dict)
    allocated_resources: dict[str, Any] = field(default_factory=dict)
    resource_reservation_id: str = ""
    capability_proof: str = ""

    def __post_init__(self) -> None:
        """Validate capability contract after initialization."""
        super().__post_init__()
        if not self.intent_contract_id:
            raise ValueError("intent_contract_id cannot be empty")
        if not self.cluster_topology:
            raise ValueError("cluster_topology cannot be empty")
        if not self.capability_proof:
            raise ValueError("capability_proof cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize capability contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "intent_contract_id": self.intent_contract_id,
                "cluster_topology": self.cluster_topology,
                "allocated_resources": self.allocated_resources,
                "resource_reservation_id": self.resource_reservation_id,
                "capability_proof": self.capability_proof,
            }
        )
        return base

    def get_cluster_type(self) -> str:
        """Get cluster type from topology.

        Returns:
            Cluster type string
        """
        return self.cluster_topology.get("cluster_type", "")

    def get_total_accelerators(self) -> int:
        """Get total number of accelerators.

        Returns:
            Total accelerator count
        """
        node_count = self.cluster_topology.get("node_count", 0)
        accel_per_node = self.cluster_topology.get("accelerators_per_node", 0)
        return node_count * accel_per_node


def create_capability_contract(
    intent_contract_id: str,
    topology: ClusterTopology,
    allocated_resources: dict[str, Any],
    capability_proof: str,
    resource_reservation_id: str = "",
) -> CapabilityContract:
    """Create a CapabilityContract from hardware binding.

    Args:
        intent_contract_id: Reference to IntentContract
        topology: ClusterTopology specification
        allocated_resources: Allocated resource details
        capability_proof: Proof of capability binding
        resource_reservation_id: Optional reservation ID

    Returns:
        Immutable CapabilityContract
    """
    content = {
        "intent_contract_id": intent_contract_id,
        "cluster_topology": topology.serialize(),
        "allocated_resources": allocated_resources,
        "resource_reservation_id": resource_reservation_id,
        "capability_proof": capability_proof,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    contract_id = generate_contract_id("CapabilityContract", content)

    return CapabilityContract(
        contract_id=contract_id,
        contract_type="CapabilityContract",
        created_at=content["created_at"],
        version=content["version"],
        intent_contract_id=content["intent_contract_id"],
        cluster_topology=content["cluster_topology"],
        allocated_resources=content["allocated_resources"],
        resource_reservation_id=content["resource_reservation_id"],
        capability_proof=content["capability_proof"],
    )

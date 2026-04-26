"""
Global Multi-Layer Infrastructure Module

Implements the physical, logical, AI, and symbolic infrastructure layers for
planet-scale operations with satellite, maritime, terrestrial, and edge nodes.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeType(Enum):
    """Types of infrastructure nodes."""

    SATELLITE = "satellite"
    MARITIME = "maritime"
    TERRESTRIAL = "terrestrial"
    EDGE = "edge"
    DATA_CENTER = "data_center"
    GATEWAY = "gateway"


class LayerType(Enum):
    """Infrastructure layer types."""

    PHYSICAL = "physical"
    LOGICAL = "logical"
    AI = "ai"
    SYMBOLIC = "symbolic"


@dataclass(frozen=True)
class PhysicalNode:
    """Represents a physical infrastructure node.

    Attributes:
        node_id: Unique node identifier
        node_type: Type of physical node
        latitude: Geographic latitude
        longitude: Geographic longitude
        altitude_km: Altitude in kilometers (for satellite nodes)
        capacity_gbps: Network capacity in Gbps
        status: Current operational status
        region: Geographic region
        metadata: Additional node metadata
    """

    node_id: str
    node_type: NodeType
    latitude: float
    longitude: float
    altitude_km: float = 0.0
    capacity_gbps: float = 10.0
    status: str = "active"
    region: str = "global"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_km": self.altitude_km,
            "capacity_gbps": self.capacity_gbps,
            "status": self.status,
            "region": self.region,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LogicalContract:
    """Blockchain-based Proof-of-Data consensus contract.

    Attributes:
        contract_id: Unique contract identifier
        contract_hash: SHA-256 hash of contract content
        data_schema: Schema for validated data
        validators: List of validator node IDs
        consensus_threshold: Required consensus percentage (0-100)
        created_at: Contract creation timestamp
        expiry_at: Contract expiration timestamp
    """

    contract_id: str
    contract_hash: str
    data_schema: dict[str, Any]
    validators: tuple[str, ...]
    consensus_threshold: int = 67
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expiry_at: str = ""

    def verify_consensus(self, votes: dict[str, bool]) -> bool:
        """Verify if consensus is reached.

        Args:
            votes: Dictionary mapping validator IDs to their votes

        Returns:
            True if consensus threshold is met
        """
        valid_votes = sum(1 for v in self.validators if votes.get(v, False))
        return (valid_votes / len(self.validators)) * 100 >= self.consensus_threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to dictionary."""
        return {
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "data_schema": self.data_schema,
            "validators": list(self.validators),
            "consensus_threshold": self.consensus_threshold,
            "created_at": self.created_at,
            "expiry_at": self.expiry_at,
        }


@dataclass
class AIGovernanceNode:
    """Decentralized AI governance node for real-time decision-making.

    Attributes:
        node_id: Unique node identifier
        model_version: AI model version
        decision_latency_ms: Average decision latency in milliseconds
        accuracy_score: Model accuracy (0.0-1.0)
        governance_scope: Scope of governance decisions
        parent_nodes: Parent nodes for hierarchical governance
        metadata: Additional node metadata
    """

    node_id: str
    model_version: str
    decision_latency_ms: float = 10.0
    accuracy_score: float = 0.95
    governance_scope: list[str] = field(default_factory=list)
    parent_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def make_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make a governance decision based on context.

        Args:
            context: Decision context including state and constraints

        Returns:
            Decision outcome with reasoning
        """
        # Deterministic decision-making placeholder
        decision_id = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]

        return {
            "decision_id": decision_id,
            "node_id": self.node_id,
            "context_hash": hashlib.sha256(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest(),
            "outcome": "approved" if context.get("priority", 0) > 0.5 else "pending",
            "confidence": self.accuracy_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "node_id": self.node_id,
            "model_version": self.model_version,
            "decision_latency_ms": self.decision_latency_ms,
            "accuracy_score": self.accuracy_score,
            "governance_scope": self.governance_scope,
            "parent_nodes": self.parent_nodes,
            "metadata": self.metadata,
        }


@dataclass
class SymbolicAttractor:
    """Recursive symbolic attractor field for decentralized governance.

    Represents governance policies as attractor basins in a symbolic state space,
    enabling emergent, self-organizing policy optimization.

    Attributes:
        attractor_id: Unique attractor identifier
        basin_depth: Depth of attractor basin (stability measure)
        convergence_rate: Rate of convergence to attractor state
        policy_vector: Numerical encoding of policy parameters
        connected_attractors: IDs of connected attractors
        stability_score: Overall stability score (0.0-1.0)
    """

    attractor_id: str
    basin_depth: float = 1.0
    convergence_rate: float = 0.1
    policy_vector: list[float] = field(default_factory=list)
    connected_attractors: list[str] = field(default_factory=list)
    stability_score: float = 0.8

    def compute_gradient(self, state: list[float]) -> list[float]:
        """Compute gradient towards attractor basin.

        Args:
            state: Current state vector

        Returns:
            Gradient vector towards attractor
        """
        if len(state) != len(self.policy_vector):
            return [0.0] * len(state)

        return [
            (target - current) * self.convergence_rate
            for current, target in zip(state, self.policy_vector)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize attractor to dictionary."""
        return {
            "attractor_id": self.attractor_id,
            "basin_depth": self.basin_depth,
            "convergence_rate": self.convergence_rate,
            "policy_vector": self.policy_vector,
            "connected_attractors": self.connected_attractors,
            "stability_score": self.stability_score,
        }


@dataclass
class InfrastructureLayer:
    """Represents a single infrastructure layer.

    Attributes:
        layer_type: Type of infrastructure layer
        nodes: Dictionary of nodes in this layer
        contracts: Dictionary of contracts (for logical layer)
        governance_nodes: Dictionary of AI governance nodes
        attractors: Dictionary of symbolic attractors
        metadata: Additional layer metadata
    """

    layer_type: LayerType
    nodes: dict[str, PhysicalNode] = field(default_factory=dict)
    contracts: dict[str, LogicalContract] = field(default_factory=dict)
    governance_nodes: dict[str, AIGovernanceNode] = field(default_factory=dict)
    attractors: dict[str, SymbolicAttractor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_physical_node(self, node: PhysicalNode) -> None:
        """Add a physical node to the layer."""
        self.nodes[node.node_id] = node

    def add_contract(self, contract: LogicalContract) -> None:
        """Add a logical contract to the layer."""
        self.contracts[contract.contract_id] = contract

    def add_governance_node(self, node: AIGovernanceNode) -> None:
        """Add an AI governance node to the layer."""
        self.governance_nodes[node.node_id] = node

    def add_attractor(self, attractor: SymbolicAttractor) -> None:
        """Add a symbolic attractor to the layer."""
        self.attractors[attractor.attractor_id] = attractor

    def get_statistics(self) -> dict[str, Any]:
        """Get layer statistics.

        Returns:
            Dictionary with layer statistics
        """
        return {
            "layer_type": self.layer_type.value,
            "physical_nodes": len(self.nodes),
            "contracts": len(self.contracts),
            "governance_nodes": len(self.governance_nodes),
            "attractors": len(self.attractors),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize layer to dictionary."""
        return {
            "layer_type": self.layer_type.value,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "contracts": {k: v.to_dict() for k, v in self.contracts.items()},
            "governance_nodes": {k: v.to_dict() for k, v in self.governance_nodes.items()},
            "attractors": {k: v.to_dict() for k, v in self.attractors.items()},
            "metadata": self.metadata,
        }


class GlobalInfrastructure:
    """Multi-layer global infrastructure manager.

    Manages the complete QRATUM infrastructure across physical, logical,
    AI, and symbolic layers with global coverage.

    Attributes:
        infrastructure_id: Unique infrastructure identifier
        layers: Dictionary of infrastructure layers
        total_capacity_tbps: Total network capacity in Tbps
        global_coverage_percent: Percentage of global coverage
        created_at: Infrastructure creation timestamp
    """

    def __init__(self, infrastructure_id: str | None = None) -> None:
        """Initialize global infrastructure.

        Args:
            infrastructure_id: Optional infrastructure ID (generated if not provided)
        """
        self.infrastructure_id = infrastructure_id or self._generate_id()
        self.layers: dict[LayerType, InfrastructureLayer] = {
            LayerType.PHYSICAL: InfrastructureLayer(LayerType.PHYSICAL),
            LayerType.LOGICAL: InfrastructureLayer(LayerType.LOGICAL),
            LayerType.AI: InfrastructureLayer(LayerType.AI),
            LayerType.SYMBOLIC: InfrastructureLayer(LayerType.SYMBOLIC),
        }
        self.total_capacity_tbps: float = 0.0
        self.global_coverage_percent: float = 0.0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._node_counter = 0

    def _generate_id(self) -> str:
        """Generate unique infrastructure ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"infra_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"

    def deploy_satellite_constellation(
        self,
        num_satellites: int = 100,
        orbit_altitude_km: float = 550.0,
        capacity_gbps: float = 20.0,
    ) -> list[PhysicalNode]:
        """Deploy a satellite constellation.

        Args:
            num_satellites: Number of satellites to deploy
            orbit_altitude_km: Orbital altitude in kilometers
            capacity_gbps: Capacity per satellite in Gbps

        Returns:
            List of deployed satellite nodes
        """
        satellites = []
        for i in range(num_satellites):
            # Distribute satellites in orbital planes
            lat = ((i % 10) - 5) * 15  # -75 to 75 degrees
            lon = ((i // 10) * 36) % 360 - 180  # -180 to 180 degrees

            node = PhysicalNode(
                node_id=f"sat_{self._node_counter:06d}",
                node_type=NodeType.SATELLITE,
                latitude=lat,
                longitude=lon,
                altitude_km=orbit_altitude_km,
                capacity_gbps=capacity_gbps,
                region="orbit",
            )
            self._node_counter += 1
            self.layers[LayerType.PHYSICAL].add_physical_node(node)
            satellites.append(node)

        self._update_capacity()
        return satellites

    def deploy_terrestrial_nodes(
        self,
        regions: list[dict[str, Any]],
        nodes_per_region: int = 10,
        capacity_gbps: float = 100.0,
    ) -> list[PhysicalNode]:
        """Deploy terrestrial infrastructure nodes.

        Args:
            regions: List of region specifications with lat/lon bounds
            nodes_per_region: Number of nodes per region
            capacity_gbps: Capacity per node in Gbps

        Returns:
            List of deployed terrestrial nodes
        """
        nodes = []
        for region in regions:
            lat_min = region.get("lat_min", -90)
            lat_max = region.get("lat_max", 90)
            lon_min = region.get("lon_min", -180)
            lon_max = region.get("lon_max", 180)
            region_name = region.get("name", "unknown")

            for i in range(nodes_per_region):
                lat = lat_min + (lat_max - lat_min) * (i / nodes_per_region)
                lon = lon_min + (lon_max - lon_min) * (i / nodes_per_region)

                node = PhysicalNode(
                    node_id=f"terr_{self._node_counter:06d}",
                    node_type=NodeType.TERRESTRIAL,
                    latitude=lat,
                    longitude=lon,
                    capacity_gbps=capacity_gbps,
                    region=region_name,
                )
                self._node_counter += 1
                self.layers[LayerType.PHYSICAL].add_physical_node(node)
                nodes.append(node)

        self._update_capacity()
        return nodes

    def deploy_edge_network(
        self,
        num_edges: int = 1000,
        capacity_gbps: float = 1.0,
    ) -> list[PhysicalNode]:
        """Deploy edge computing nodes globally.

        Args:
            num_edges: Number of edge nodes to deploy
            capacity_gbps: Capacity per edge node in Gbps

        Returns:
            List of deployed edge nodes
        """
        edges = []
        for i in range(num_edges):
            # Distribute edges globally
            lat = ((i * 7) % 180) - 90
            lon = ((i * 13) % 360) - 180

            node = PhysicalNode(
                node_id=f"edge_{self._node_counter:06d}",
                node_type=NodeType.EDGE,
                latitude=lat,
                longitude=lon,
                capacity_gbps=capacity_gbps,
                region="edge",
            )
            self._node_counter += 1
            self.layers[LayerType.PHYSICAL].add_physical_node(node)
            edges.append(node)

        self._update_capacity()
        return edges

    def deploy_maritime_nodes(
        self,
        num_nodes: int = 50,
        capacity_gbps: float = 10.0,
    ) -> list[PhysicalNode]:
        """Deploy maritime infrastructure nodes.

        Args:
            num_nodes: Number of maritime nodes to deploy
            capacity_gbps: Capacity per node in Gbps

        Returns:
            List of deployed maritime nodes
        """
        nodes = []
        for i in range(num_nodes):
            # Distribute along major shipping lanes
            lat = ((i * 3) % 120) - 60
            lon = ((i * 23) % 360) - 180

            node = PhysicalNode(
                node_id=f"mar_{self._node_counter:06d}",
                node_type=NodeType.MARITIME,
                latitude=lat,
                longitude=lon,
                capacity_gbps=capacity_gbps,
                region="maritime",
            )
            self._node_counter += 1
            self.layers[LayerType.PHYSICAL].add_physical_node(node)
            nodes.append(node)

        self._update_capacity()
        return nodes

    def create_data_contract(
        self,
        data_schema: dict[str, Any],
        validators: list[str],
        consensus_threshold: int = 67,
    ) -> LogicalContract:
        """Create a Proof-of-Data consensus contract.

        Args:
            data_schema: Schema for data validation
            validators: List of validator node IDs
            consensus_threshold: Required consensus percentage

        Returns:
            Created logical contract
        """
        content = json.dumps({"schema": data_schema, "validators": validators}, sort_keys=True)
        contract_hash = hashlib.sha256(content.encode()).hexdigest()
        contract_id = f"contract_{contract_hash[:12]}"

        contract = LogicalContract(
            contract_id=contract_id,
            contract_hash=contract_hash,
            data_schema=data_schema,
            validators=tuple(validators),
            consensus_threshold=consensus_threshold,
        )

        self.layers[LayerType.LOGICAL].add_contract(contract)
        return contract

    def deploy_ai_governance(
        self,
        num_nodes: int = 10,
        governance_scope: list[str] | None = None,
    ) -> list[AIGovernanceNode]:
        """Deploy AI governance nodes.

        Args:
            num_nodes: Number of governance nodes
            governance_scope: Scope of governance decisions

        Returns:
            List of deployed governance nodes
        """
        scope = governance_scope or ["policy", "optimization", "security"]
        nodes = []

        for i in range(num_nodes):
            node = AIGovernanceNode(
                node_id=f"ai_gov_{i:04d}",
                model_version="1.0.0",
                governance_scope=scope,
            )
            self.layers[LayerType.AI].add_governance_node(node)
            nodes.append(node)

        return nodes

    def create_symbolic_attractor(
        self,
        policy_vector: list[float],
        basin_depth: float = 1.0,
        convergence_rate: float = 0.1,
    ) -> SymbolicAttractor:
        """Create a symbolic attractor for policy optimization.

        Args:
            policy_vector: Numerical encoding of policy
            basin_depth: Depth of attractor basin
            convergence_rate: Rate of convergence

        Returns:
            Created symbolic attractor
        """
        attractor_id = f"attr_{hashlib.sha256(str(policy_vector).encode()).hexdigest()[:8]}"

        attractor = SymbolicAttractor(
            attractor_id=attractor_id,
            basin_depth=basin_depth,
            convergence_rate=convergence_rate,
            policy_vector=policy_vector,
        )

        self.layers[LayerType.SYMBOLIC].add_attractor(attractor)
        return attractor

    def _update_capacity(self) -> None:
        """Update total capacity and coverage metrics."""
        physical_layer = self.layers[LayerType.PHYSICAL]
        total_gbps = sum(node.capacity_gbps for node in physical_layer.nodes.values())
        self.total_capacity_tbps = total_gbps / 1000

        # Estimate coverage based on node distribution
        unique_regions = set()
        for node in physical_layer.nodes.values():
            lat_region = int(node.latitude / 30)
            lon_region = int(node.longitude / 30)
            unique_regions.add((lat_region, lon_region))

        total_regions = 12 * 6  # 30-degree cells
        self.global_coverage_percent = min(100, (len(unique_regions) / total_regions) * 100)

    def get_statistics(self) -> dict[str, Any]:
        """Get infrastructure statistics.

        Returns:
            Dictionary with infrastructure statistics
        """
        physical_stats = self.layers[LayerType.PHYSICAL].get_statistics()
        logical_stats = self.layers[LayerType.LOGICAL].get_statistics()
        ai_stats = self.layers[LayerType.AI].get_statistics()
        symbolic_stats = self.layers[LayerType.SYMBOLIC].get_statistics()

        return {
            "infrastructure_id": self.infrastructure_id,
            "total_capacity_tbps": self.total_capacity_tbps,
            "global_coverage_percent": self.global_coverage_percent,
            "created_at": self.created_at,
            "layers": {
                "physical": physical_stats,
                "logical": logical_stats,
                "ai": ai_stats,
                "symbolic": symbolic_stats,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize infrastructure to dictionary."""
        return {
            "infrastructure_id": self.infrastructure_id,
            "total_capacity_tbps": self.total_capacity_tbps,
            "global_coverage_percent": self.global_coverage_percent,
            "created_at": self.created_at,
            "layers": {k.value: v.to_dict() for k, v in self.layers.items()},
        }

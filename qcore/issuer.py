"""Q-Core Contract Issuer - Issues All 4 Contract Types.

This module implements the contract issuer, which generates all 4 required
contracts (Intent, Capability, Temporal, Event) from an authorized intent.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts import (
    CapabilityContract,
    ClusterTopology,
    EventContract,
    IntentContract,
    TemporalContract,
    create_capability_contract,
    create_event_contract,
    create_intent_contract,
    create_temporal_contract,
)
from qcore.authority import AuthorizationResult
from qcore.resolver import CapabilityResolver, ResolvedCapability
from qil.ast import Intent


@dataclass
class ContractBundle:
    """Bundle of all contracts for an intent.

    Attributes:
        intent_contract: IntentContract
        capability_contract: CapabilityContract
        temporal_contract: TemporalContract
        event_contract: EventContract
    """

    intent_contract: IntentContract
    capability_contract: CapabilityContract
    temporal_contract: TemporalContract
    event_contract: EventContract


class ContractIssuer:
    """Issues immutable contracts from authorized intents."""

    def __init__(
        self,
        resolver: CapabilityResolver | None = None,
        cluster_registry: dict[str, Any] | None = None,
    ) -> None:
        """Initialize contract issuer.

        Args:
            resolver: Optional CapabilityResolver instance
            cluster_registry: Optional cluster registry for topology
        """
        self.resolver = resolver or CapabilityResolver()
        self.cluster_registry = cluster_registry or {}

    def issue_contracts(self, intent: Intent, authorization: AuthorizationResult) -> ContractBundle:
        """Issue all 4 contracts from an authorized intent.

        Args:
            intent: Authorized Intent
            authorization: Authorization result with proof

        Returns:
            ContractBundle with all 4 contracts

        Raises:
            ValueError: If authorization is invalid
        """
        if not authorization.authorized:
            raise ValueError(
                f"Cannot issue contracts for unauthorized intent: {authorization.reason}"
            )

        # 1. Create IntentContract
        resolved_capabilities = self.resolver.resolve_capabilities(intent)
        capability_names = [rc.name for rc in resolved_capabilities]

        intent_contract = create_intent_contract(
            intent=intent,
            authorization_proof=authorization.proof,
            resolved_capabilities=capability_names,
        )

        # 2. Create CapabilityContract
        capability_contract = self._create_capability_contract(
            intent_contract, resolved_capabilities
        )

        # 3. Create TemporalContract
        temporal_contract = self._create_temporal_contract(intent_contract, intent)

        # 4. Create EventContract
        event_contract = self._create_event_contract(intent_contract)

        return ContractBundle(
            intent_contract=intent_contract,
            capability_contract=capability_contract,
            temporal_contract=temporal_contract,
            event_contract=event_contract,
        )

    def _create_capability_contract(
        self,
        intent_contract: IntentContract,
        resolved_capabilities: list[ResolvedCapability],
    ) -> CapabilityContract:
        """Create CapabilityContract from resolved capabilities.

        Args:
            intent_contract: IntentContract reference
            resolved_capabilities: List of resolved capabilities

        Returns:
            CapabilityContract
        """
        # Select primary cluster type (first resolved capability)
        if not resolved_capabilities:
            raise ValueError("No capabilities resolved for intent")

        primary_capability = resolved_capabilities[0]
        cluster_type = primary_capability.cluster_type

        # Create ClusterTopology
        topology = self._create_cluster_topology(
            cluster_type, primary_capability.resource_requirements
        )

        # Build allocated resources
        allocated_resources = {
            "capabilities": [rc.name for rc in resolved_capabilities],
            "primary_cluster": cluster_type,
            "resource_requirements": {
                rc.name: rc.resource_requirements for rc in resolved_capabilities
            },
        }

        # Generate capability proof
        capability_proof = f"CAP_{intent_contract.intent_name}_{cluster_type}"

        return create_capability_contract(
            intent_contract_id=intent_contract.contract_id,
            topology=topology,
            allocated_resources=allocated_resources,
            capability_proof=capability_proof,
            resource_reservation_id=f"RES_{intent_contract.contract_id[:16]}",
        )

    def _create_cluster_topology(
        self, cluster_type: str, requirements: dict[str, Any]
    ) -> ClusterTopology:
        """Create ClusterTopology for a cluster type.

        Args:
            cluster_type: Type of cluster
            requirements: Resource requirements

        Returns:
            ClusterTopology
        """
        # Default topology configurations
        topologies = {
            "GB200": ClusterTopology(
                cluster_type="GB200",
                node_count=requirements.get("nodes", 1),
                accelerators_per_node=8,  # 4x Grace + 4x Blackwell
                memory_per_node_gb=1536,  # 192GB per GPU * 8
                interconnect="NVLink5",
                metadata={"vram_per_gpu_gb": 192, "architecture": "Blackwell"},
            ),
            "MI300X": ClusterTopology(
                cluster_type="MI300X",
                node_count=requirements.get("nodes", 1),
                accelerators_per_node=8,
                memory_per_node_gb=1536,  # 192GB per GPU * 8
                interconnect="Infinity Fabric",
                metadata={"vram_per_gpu_gb": 192, "architecture": "CDNA3"},
            ),
            "QPU": ClusterTopology(
                cluster_type="QPU",
                node_count=1,
                accelerators_per_node=requirements.get("qubits", 100),
                memory_per_node_gb=1.0,  # Minimal classical control memory
                interconnect="Quantum",
                metadata={
                    "qubits": requirements.get("qubits", 100),
                    "depth": requirements.get("depth", 100),
                },
            ),
            "CEREBRAS": ClusterTopology(
                cluster_type="CEREBRAS",
                node_count=1,
                accelerators_per_node=1,  # Wafer-scale engine
                memory_per_node_gb=44000,  # 44GB on-wafer SRAM
                interconnect="On-Wafer",
                metadata={"cores": 900000, "wafer_scale": True},
            ),
            "IPU": ClusterTopology(
                cluster_type="IPU",
                node_count=1,
                accelerators_per_node=requirements.get("ipus", 4),
                memory_per_node_gb=3.6,  # 900MB per IPU * 4
                interconnect="IPU-Fabric",
                metadata={"tiles_per_ipu": 1472},
            ),
            "GAUDI3": ClusterTopology(
                cluster_type="GAUDI3",
                node_count=requirements.get("nodes", 1),
                accelerators_per_node=8,
                memory_per_node_gb=1024,  # 128GB per accelerator * 8
                interconnect="Gaudi-Fabric",
                metadata={"hbm_per_accelerator_gb": 128},
            ),
            "CPU": ClusterTopology(
                cluster_type="CPU",
                node_count=requirements.get("nodes", 1),
                accelerators_per_node=requirements.get("cores", 64),
                memory_per_node_gb=512,
                interconnect="PCIe",
                metadata={"cores": requirements.get("cores", 64)},
            ),
        }

        return topologies.get(
            cluster_type,
            ClusterTopology(
                cluster_type=cluster_type,
                node_count=1,
                accelerators_per_node=1,
                memory_per_node_gb=64,
                interconnect="Unknown",
            ),
        )

    def _create_temporal_contract(
        self, intent_contract: IntentContract, intent: Intent
    ) -> TemporalContract:
        """Create TemporalContract from intent time specifications.

        Args:
            intent_contract: IntentContract reference
            intent: Original Intent

        Returns:
            TemporalContract
        """
        deadline_seconds = 0.0
        budget_seconds = 0.0

        for time_spec in intent.time_specs:
            seconds = time_spec.to_seconds()
            if time_spec.key == "deadline":
                deadline_seconds = seconds
            elif time_spec.key == "budget":
                budget_seconds = seconds

        # Default deadline if not specified
        if deadline_seconds == 0.0 and budget_seconds == 0.0:
            deadline_seconds = 3600.0  # 1 hour default

        return create_temporal_contract(
            intent_contract_id=intent_contract.contract_id,
            deadline_seconds=deadline_seconds,
            budget_seconds=budget_seconds,
            rollback_authorized=True,  # Default to authorized
            temporal_proof=f"TIME_{intent_contract.intent_name}",
        )

    def _create_event_contract(self, intent_contract: IntentContract) -> EventContract:
        """Create EventContract with expected event sequence.

        Args:
            intent_contract: IntentContract reference

        Returns:
            EventContract
        """
        # Define expected event sequence
        expected_events = [
            "IntentReceived",
            "IntentAuthorized",
            "ContractsIssued",
            "CapabilityBound",
            "ExecutionStarted",
            "ExecutionCompleted",
            "AuditLogged",
        ]

        # Define causal chains
        causal_chains = [
            {"source": "IntentReceived", "target": "IntentAuthorized"},
            {"source": "IntentAuthorized", "target": "ContractsIssued"},
            {"source": "ContractsIssued", "target": "CapabilityBound"},
            {"source": "CapabilityBound", "target": "ExecutionStarted"},
            {"source": "ExecutionStarted", "target": "ExecutionCompleted"},
            {"source": "ExecutionCompleted", "target": "AuditLogged"},
        ]

        return create_event_contract(
            intent_contract_id=intent_contract.contract_id,
            expected_events=expected_events,
            causal_chains=causal_chains,
            event_proof=f"EVENT_{intent_contract.intent_name}",
        )

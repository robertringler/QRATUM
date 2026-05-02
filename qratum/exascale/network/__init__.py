"""
AetherFabric-X Deterministic Interconnect

A novel QRATUM innovation providing deterministic, hardware-verified
communication at exascale with < 500 ns any-to-any latency.

Components:
- aetherfabric: Core protocol implementation
- dsr: Deterministic Source Routing
- merkle_nic: Hardware Merkle verification at line rate
- flow_control: Credit-based flow control (no adaptive routing)
"""

from .aetherfabric import (AetherFabricX, NetworkTopology,
                           create_exascale_network)
from .dsr import (DSRController, Route, RoutingStrategy, RoutingTable,
                  create_dsr_controller)
from .flow_control import (CreditCounter, FlowControlLink, FlowControlManager,
                           FlowState, create_flow_control_link,
                           create_flow_control_manager)
from .merkle_nic import (MerkleNIC, MerkleProof, MerkleTree, Packet,
                         create_merkle_nic, create_packet)

__all__ = [
    "AetherFabricX",
    "NetworkTopology",
    "create_exascale_network",
    "DSRController",
    "RoutingTable",
    "Route",
    "RoutingStrategy",
    "create_dsr_controller",
    "MerkleNIC",
    "MerkleTree",
    "Packet",
    "MerkleProof",
    "create_merkle_nic",
    "create_packet",
    "FlowControlManager",
    "FlowControlLink",
    "CreditCounter",
    "FlowState",
    "create_flow_control_link",
    "create_flow_control_manager",
]

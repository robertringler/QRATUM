"""
QRATUM Platform Core Infrastructure

Provides the foundational components for deterministic, auditable execution:
- PlatformIntent: Immutable computation requests
- PlatformContract: Authorized, signed execution contracts
- MerkleEventChain: Thread-safe, cryptographically-verified event log
- PlatformOrchestrator: Routes intents to appropriate vertical modules
"""

from .core import (FATAL_INVARIANTS, ContractStatus, Event, EventType,
                   PlatformContract, PlatformIntent,
                   create_contract_from_intent, create_event)
from .event_chain import MerkleEventChain
from .orchestrator import PlatformOrchestrator
from .substrates import ComputeSubstrate, SubstrateSelector

__all__ = [
    "PlatformIntent",
    "PlatformContract",
    "Event",
    "EventType",
    "ContractStatus",
    "create_contract_from_intent",
    "create_event",
    "FATAL_INVARIANTS",
    "MerkleEventChain",
    "PlatformOrchestrator",
    "ComputeSubstrate",
    "SubstrateSelector",
]

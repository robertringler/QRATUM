"""
QRATUM Platform Core - Immutable Contracts and Events

This module implements the core abstractions that enforce QRATUM's 8 fatal invariants.
All computations start with PlatformIntent, are authorized into PlatformContract,
and emit immutable Events to the MerkleEventChain.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

# 8 Fatal Invariants - violations terminate execution
FATAL_INVARIANTS = [
    "1. Every computation MUST start with a PlatformIntent",
    "2. Q-Core authorization MUST create an immutable PlatformContract",
    "3. Contract hash MUST be deterministic and reproducible",
    "4. All execution MUST emit Events to MerkleEventChain",
    "5. MerkleEventChain MUST maintain cryptographic integrity",
    "6. Event replay MUST produce identical results (determinism)",
    "7. No in-place mutation of frozen state",
    "8. Contract signature MUST be verified before execution",
]


class ContractStatus(Enum):
    """Contract lifecycle states"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    """Event types for execution tracking"""
    CONTRACT_CREATED = "contract_created"
    CONTRACT_AUTHORIZED = "contract_authorized"
    EXECUTION_STARTED = "execution_started"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    SAFETY_VIOLATION = "safety_violation"
    INVARIANT_VIOLATION = "invariant_violation"


@dataclass(frozen=True)
class PlatformIntent:
    """
    Immutable representation of a user's computation request.
    
    This is the entry point for all QRATUM computations. Every intent
    must specify the vertical module, task, and parameters.
    
    Attributes:
        vertical: Target vertical module (e.g., "JURIS", "VITRA")
        task: Specific task within the vertical (e.g., "analyze_contract")
        parameters: Task-specific parameters (immutable)
        requester_id: Identity of the requesting entity
        timestamp: UTC timestamp of intent creation
        intent_id: Unique identifier (derived from content hash)
    """
    vertical: str
    task: str
    parameters: Dict[str, Any]
    requester_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    intent_id: str = field(default="", init=False)

    def __post_init__(self):
        """Generate deterministic intent_id from content hash"""
        content = {
            "vertical": self.vertical,
            "task": self.task,
            "parameters": self.parameters,
            "requester_id": self.requester_id,
            "timestamp": self.timestamp,
        }
        content_str = json.dumps(content, sort_keys=True)
        intent_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        object.__setattr__(self, "intent_id", f"intent_{intent_hash}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "intent_id": self.intent_id,
            "vertical": self.vertical,
            "task": self.task,
            "parameters": self.parameters,
            "requester_id": self.requester_id,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class PlatformContract:
    """
    Immutable, cryptographically-signed execution contract.
    
    Created by Q-Core authorization from a PlatformIntent. The contract
    is the binding agreement to execute a computation with specific
    parameters and guarantees.
    
    Attributes:
        intent: The original PlatformIntent
        contract_id: Unique contract identifier
        authorized_by: Q-Core entity that authorized the contract
        status: Current contract status
        signature: Cryptographic signature (SHA-256 of contract content)
        created_at: UTC timestamp of contract creation
        authorized_at: UTC timestamp of authorization
    """
    intent: PlatformIntent
    contract_id: str
    authorized_by: str
    status: ContractStatus
    created_at: str
    authorized_at: str
    signature: str = field(default="", init=False)

    def __post_init__(self):
        """Generate deterministic contract signature"""
        content = {
            "intent": self.intent.to_dict(),
            "contract_id": self.contract_id,
            "authorized_by": self.authorized_by,
            "status": self.status.value,
            "created_at": self.created_at,
            "authorized_at": self.authorized_at,
        }
        content_str = json.dumps(content, sort_keys=True)
        sig = hashlib.sha256(content_str.encode()).hexdigest()
        object.__setattr__(self, "signature", sig)

    def verify_signature(self) -> bool:
        """Verify contract signature integrity"""
        content = {
            "intent": self.intent.to_dict(),
            "contract_id": self.contract_id,
            "authorized_by": self.authorized_by,
            "status": self.status.value,
            "created_at": self.created_at,
            "authorized_at": self.authorized_at,
        }
        content_str = json.dumps(content, sort_keys=True)
        expected_sig = hashlib.sha256(content_str.encode()).hexdigest()
        return self.signature == expected_sig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "contract_id": self.contract_id,
            "intent": self.intent.to_dict(),
            "authorized_by": self.authorized_by,
            "status": self.status.value,
            "signature": self.signature,
            "created_at": self.created_at,
            "authorized_at": self.authorized_at,
        }


@dataclass(frozen=True)
class Event:
    """
    Immutable event in the execution chain.
    
    Every significant action during contract execution emits an Event.
    Events are appended to the MerkleEventChain for auditability.
    
    Attributes:
        event_id: Unique event identifier
        event_type: Type of event (from EventType enum)
        contract_id: Associated contract ID
        timestamp: UTC timestamp of event
        data: Event-specific data payload
        emitter: Component that emitted the event
    """
    event_id: str
    event_type: EventType
    contract_id: str
    timestamp: str
    data: Dict[str, Any]
    emitter: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "contract_id": self.contract_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "emitter": self.emitter,
        }


def create_contract_from_intent(
    intent: PlatformIntent,
    authorized_by: str = "Q-Core"
) -> PlatformContract:
    """
    Create an authorized PlatformContract from a PlatformIntent.
    
    This simulates the Q-Core authorization process. In production,
    this would involve cryptographic key signing and policy validation.
    
    Args:
        intent: The PlatformIntent to authorize
        authorized_by: The authorizing entity (default: "Q-Core")
    
    Returns:
        Authorized PlatformContract
    """
    now = datetime.now(timezone.utc).isoformat()
    contract_id = f"contract_{intent.intent_id}_{hashlib.sha256(now.encode()).hexdigest()[:8]}"

    contract = PlatformContract(
        intent=intent,
        contract_id=contract_id,
        authorized_by=authorized_by,
        status=ContractStatus.AUTHORIZED,
        created_at=now,
        authorized_at=now,
    )

    # Verify signature was generated correctly
    if not contract.verify_signature():
        raise RuntimeError("FATAL: Contract signature verification failed (Invariant #8)")

    return contract


def create_event(
    event_type: EventType,
    contract_id: str,
    data: Dict[str, Any],
    emitter: str
) -> Event:
    """
    Create an immutable Event.
    
    Args:
        event_type: Type of event
        contract_id: Associated contract ID
        data: Event-specific data
        emitter: Component emitting the event
    
    Returns:
        Immutable Event
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    event_id = f"event_{hashlib.sha256(f'{contract_id}_{timestamp}_{emitter}'.encode()).hexdigest()[:16]}"

    return Event(
        event_id=event_id,
        event_type=event_type,
        contract_id=contract_id,
        timestamp=timestamp,
        data=data,
        emitter=emitter,
    )

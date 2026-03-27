"""Core platform classes for QRATUM Sovereign AI Platform v2.0.

This module defines the foundational types, contracts, and base classes
for deterministic, auditable execution of vertical AI modules.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerticalModule(Enum):
    """Enumeration of all 14 vertical AI modules in QRATUM platform."""

    JURIS = "juris"  # Legal AI
    VITRA = "vitra"  # Bioinformatics
    ECORA = "ecora"  # Climate & Energy
    CAPRA = "capra"  # Financial Risk
    SENTRA = "sentra"  # Aerospace & Defense
    NEURA = "neura"  # Neuroscience & BCI
    FLUXA = "fluxa"  # Supply Chain
    SPECTRA = "spectra"  # Spectrum Management
    AEGIS = "aegis"  # Cybersecurity
    LOGOS = "logos"  # Education & Training
    SYNTHOS = "synthos"  # Materials Science
    TERAGON = "teragon"  # Geospatial Intelligence
    HELIX = "helix"  # Genomic Medicine
    NEXUS = "nexus"  # Cross-domain Intelligence


class ComputeSubstrate(Enum):
    """Enumeration of compute substrates available in Frankenstein Cluster."""

    GB200 = "gb200"  # NVIDIA GB200 (Grace-Blackwell)
    MI300X = "mi300x"  # AMD MI300X
    CEREBRAS = "cerebras"  # Cerebras CS-3
    IPU = "ipu"  # Graphcore IPU
    GAUDI = "gaudi"  # Intel Gaudi 3
    QPU = "qpu"  # Quantum Processing Unit
    CPU = "cpu"  # Standard CPU


class SafetyViolation(Exception):
    """Exception raised when prohibited use is detected."""

    pass


@dataclass(frozen=True)
class PlatformIntent:
    """Universal intent structure for QRATUM platform.

    Immutable structure representing a user's intent to execute
    a specific vertical module with parameters.
    """

    vertical: VerticalModule
    operation: str
    parameters: Dict[str, Any]
    user_id: str
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary for serialization."""
        return {
            "vertical": self.vertical.value,
            "operation": self.operation,
            "parameters": self.parameters,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize intent to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of intent."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True)
class PlatformContract:
    """Immutable execution contract for QRADLE.

    Represents a binding contract between user intent and platform execution.
    Once created, cannot be modified (enforces contract immutability invariant).
    """

    intent: PlatformIntent
    contract_id: str
    substrate: ComputeSubstrate
    estimated_cost: float
    estimated_duration: float
    safety_checks_passed: bool
    compliance_flags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for serialization."""
        return {
            "intent": self.intent.to_dict(),
            "contract_id": self.contract_id,
            "substrate": self.substrate.value,
            "estimated_cost": self.estimated_cost,
            "estimated_duration": self.estimated_duration,
            "safety_checks_passed": self.safety_checks_passed,
            "compliance_flags": self.compliance_flags,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        """Serialize contract to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of contract."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class ExecutionEvent:
    """Event for Merkle chain audit trail.

    Records a single event in the execution lifecycle of a contract.
    """

    event_type: str
    contract_id: str
    timestamp: float
    data: Dict[str, Any]
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def __post_init__(self):
        """Compute event hash after initialization."""
        if self.event_hash is None:
            self.event_hash = self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "contract_id": self.contract_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }

    def to_json(self) -> str:
        """Serialize event to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class MerkleEventChain:
    """Append-only audit trail using Merkle chain.

    Provides hash-chain integrity and causal traceability for all events.
    """

    def __init__(self):
        self.events: List[ExecutionEvent] = []
        self.last_hash: Optional[str] = None

    def append(self, event_type: str, contract_id: str, data: Dict[str, Any]) -> ExecutionEvent:
        """Append a new event to the chain.

        Args:
            event_type: Type of event (e.g., "contract_created", "execution_started")
            contract_id: ID of the contract this event relates to
            data: Event-specific data

        Returns:
            The created ExecutionEvent
        """
        event = ExecutionEvent(
            event_type=event_type,
            contract_id=contract_id,
            timestamp=time.time(),
            data=data,
            previous_hash=self.last_hash,
        )
        self.events.append(event)
        self.last_hash = event.event_hash
        return event

    def verify_chain(self) -> bool:
        """Verify integrity of the entire event chain.

        Returns:
            True if chain is valid, False otherwise
        """
        if not self.events:
            return True

        prev_hash = None
        for event in self.events:
            if event.previous_hash != prev_hash:
                return False
            if event.event_hash != event.compute_hash():
                return False
            prev_hash = event.event_hash

        return True

    def get_events_for_contract(self, contract_id: str) -> List[ExecutionEvent]:
        """Get all events for a specific contract.

        Args:
            contract_id: ID of the contract

        Returns:
            List of events for the contract
        """
        return [e for e in self.events if e.contract_id == contract_id]


class VerticalModuleBase(ABC):
    """Abstract base class for all vertical modules.

    Enforces QRATUM fatal invariants and provides common functionality.
    """

    MODULE_NAME: str = "base"
    MODULE_VERSION: str = "0.0.0"
    SAFETY_DISCLAIMER: str = "Base module - not for production use"
    PROHIBITED_USES: List[str] = []

    def __init__(self):
        self.event_chain = MerkleEventChain()

    @abstractmethod
    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute the module's primary operation.

        Args:
            contract: Immutable execution contract

        Returns:
            Results dictionary

        Raises:
            SafetyViolation: If prohibited use is detected
        """
        pass

    @abstractmethod
    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Determine optimal compute substrate for operation.

        Args:
            operation: Operation to perform
            parameters: Operation parameters

        Returns:
            Optimal compute substrate
        """
        pass

    def check_safety(self, intent: PlatformIntent) -> bool:
        """Check if intent violates safety policies.

        Args:
            intent: User intent to check

        Returns:
            True if safe, False otherwise

        Raises:
            SafetyViolation: If prohibited use is detected
        """
        # Check for prohibited keywords in operation and parameters
        intent_str = json.dumps(intent.to_dict()).lower()
        for prohibited in self.PROHIBITED_USES:
            if prohibited.lower() in intent_str:
                raise SafetyViolation(
                    f"Prohibited use detected: {prohibited}. "
                    f"See {self.MODULE_NAME} safety disclaimer."
                )
        return True

    def emit_event(self, event_type: str, contract_id: str, data: Dict[str, Any]) -> ExecutionEvent:
        """Emit an event to the audit trail.

        Args:
            event_type: Type of event
            contract_id: Contract ID
            data: Event data

        Returns:
            The emitted event
        """
        return self.event_chain.append(event_type, contract_id, data)

    def get_disclaimer(self) -> str:
        """Get safety disclaimer for this module.

        Returns:
            Safety disclaimer text
        """
        return self.SAFETY_DISCLAIMER


class QRATUMPlatform:
    """Main orchestrator for QRATUM Sovereign AI Platform.

    Coordinates vertical modules, enforces invariants, and manages execution.
    """

    def __init__(self):
        self.event_chain = MerkleEventChain()
        self.modules: Dict[VerticalModule, VerticalModuleBase] = {}
        self.contracts: Dict[str, PlatformContract] = {}

    def register_module(self, vertical: VerticalModule, module: VerticalModuleBase):
        """Register a vertical module with the platform.

        Args:
            vertical: Vertical module type
            module: Module instance
        """
        self.modules[vertical] = module

    def create_contract(self, intent: PlatformIntent) -> PlatformContract:
        """Create an immutable execution contract from intent.

        Args:
            intent: User intent

        Returns:
            Created contract

        Raises:
            SafetyViolation: If safety checks fail
            ValueError: If vertical module not registered
        """
        # Verify module is registered
        if intent.vertical not in self.modules:
            raise ValueError(f"Vertical module {intent.vertical} not registered")

        module = self.modules[intent.vertical]

        # Perform safety checks
        module.check_safety(intent)

        # Get optimal substrate
        substrate = module.get_optimal_substrate(intent.operation, intent.parameters)

        # Create contract
        contract_id = f"{intent.vertical.value}_{intent.compute_hash()[:16]}"
        contract = PlatformContract(
            intent=intent,
            contract_id=contract_id,
            substrate=substrate,
            estimated_cost=0.0,  # TODO: Implement cost estimation
            estimated_duration=0.0,  # TODO: Implement duration estimation
            safety_checks_passed=True,
            compliance_flags=[],
        )

        # Store contract
        self.contracts[contract_id] = contract

        # Emit event
        self.event_chain.append(
            "contract_created",
            contract_id,
            {"intent_hash": intent.compute_hash(), "substrate": substrate.value},
        )

        return contract

    def execute_contract(self, contract_id: str) -> Dict[str, Any]:
        """Execute a contract on the appropriate vertical module.

        Args:
            contract_id: ID of contract to execute

        Returns:
            Execution results

        Raises:
            ValueError: If contract not found or module not registered
        """
        if contract_id not in self.contracts:
            raise ValueError(f"Contract {contract_id} not found")

        contract = self.contracts[contract_id]
        vertical = contract.intent.vertical

        if vertical not in self.modules:
            raise ValueError(f"Vertical module {vertical} not registered")

        module = self.modules[vertical]

        # Emit execution start event
        self.event_chain.append("execution_started", contract_id, {"timestamp": time.time()})

        try:
            # Execute the module
            results = module.execute(contract)

            # Emit execution complete event
            self.event_chain.append(
                "execution_completed",
                contract_id,
                {"timestamp": time.time(), "success": True},
            )

            return results

        except Exception as e:
            # Emit execution failed event
            self.event_chain.append(
                "execution_failed",
                contract_id,
                {"timestamp": time.time(), "error": str(e)},
            )
            raise

    def verify_integrity(self) -> bool:
        """Verify integrity of platform event chain.

        Returns:
            True if chain is valid, False otherwise
        """
        return self.event_chain.verify_chain()

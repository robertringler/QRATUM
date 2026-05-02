"""
Contract system for QRADLE.

Contracts are immutable units of work that specify inputs, operations,
and expected outputs. They can be rolled back to any previous state.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ContractStatus(Enum):
    """Status of a contract execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Contract:
    """Immutable contract for deterministic execution.

    Once created, a contract cannot be modified. This enforces
    the contract immutability invariant.
    """

    contract_id: str
    operation: str
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    user_id: str = ""
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for serialization."""
        return {
            "contract_id": self.contract_id,
            "operation": self.operation,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize contract to deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of contract."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class ContractExecution:
    """Result of contract execution."""

    contract: Contract
    status: ContractStatus
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    proof_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "contract": self.contract.to_dict(),
            "status": self.status.value,
            "outputs": self.outputs,
            "error": self.error,
            "execution_time": self.execution_time,
            "proof_hash": self.proof_hash,
        }

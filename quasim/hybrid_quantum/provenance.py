"""Quantum Provenance Tracking for QRATUM.

This module implements full provenance tracking for quantum executions,
enabling verification that outputs correspond to declared inputs.

Key features:
- Deterministic hashing of circuit inputs
- Execution timeline tracking
- Cryptographic provenance records
- Integration with QRATUM trust invariants
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProvenanceStatus(Enum):
    """Status of provenance record."""

    PENDING = "pending"
    RECORDED = "recorded"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class ProvenanceRecord:
    """Immutable record of quantum execution provenance.

    This record captures all information needed to verify that
    a quantum execution result corresponds to its declared inputs.

    Attributes:
        record_id: Unique identifier for this provenance record
        execution_id: ID of the quantum execution
        input_hash: SHA-256 hash of circuit and parameters
        output_hash: SHA-256 hash of execution results
        timestamp: When the execution occurred
        backend_provider: Quantum backend provider used
        backend_device: Specific device or simulator name
        circuit_depth: Depth of the executed circuit
        circuit_gates: Number of gates in the circuit
        shots: Number of measurement shots
        status: Current status of the provenance record
        metadata: Additional metadata for tracking
        parent_record_id: ID of parent record (for rollback chains)
    """

    record_id: str
    execution_id: str
    input_hash: str
    output_hash: str = ""
    timestamp: str = ""
    backend_provider: str = ""
    backend_device: str = ""
    circuit_depth: int = 0
    circuit_gates: int = 0
    shots: int = 0
    status: ProvenanceStatus = ProvenanceStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_record_id: str | None = None

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def compute_signature(self) -> str:
        """Compute cryptographic signature of this record.

        Returns:
            SHA-256 hex digest of record contents
        """
        content = (
            f"{self.record_id}|{self.execution_id}|{self.input_hash}|"
            f"{self.output_hash}|{self.timestamp}|{self.backend_provider}|"
            f"{self.backend_device}|{self.shots}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all record fields
        """
        return {
            "record_id": self.record_id,
            "execution_id": self.execution_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp,
            "backend_provider": self.backend_provider,
            "backend_device": self.backend_device,
            "circuit_depth": self.circuit_depth,
            "circuit_gates": self.circuit_gates,
            "shots": self.shots,
            "status": self.status.value,
            "metadata": self.metadata,
            "parent_record_id": self.parent_record_id,
            "signature": self.compute_signature(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceRecord":
        """Create record from dictionary.

        Args:
            data: Dictionary with record fields

        Returns:
            ProvenanceRecord instance
        """
        return cls(
            record_id=data["record_id"],
            execution_id=data["execution_id"],
            input_hash=data["input_hash"],
            output_hash=data.get("output_hash", ""),
            timestamp=data.get("timestamp", ""),
            backend_provider=data.get("backend_provider", ""),
            backend_device=data.get("backend_device", ""),
            circuit_depth=data.get("circuit_depth", 0),
            circuit_gates=data.get("circuit_gates", 0),
            shots=data.get("shots", 0),
            status=ProvenanceStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
            parent_record_id=data.get("parent_record_id"),
        )


class QuantumProvenanceWrapper:
    """Wrapper that adds provenance tracking to quantum executions.

    This wrapper ensures all quantum executions are tracked with full
    provenance information, enabling verification and rollback.

    Example:
        >>> backend = IBMHybridBackend(config)
        >>> wrapper = QuantumProvenanceWrapper(backend)
        >>> result, record = wrapper.execute_with_provenance(circuit, shots=1024)
        >>> assert record.status == ProvenanceStatus.RECORDED
    """

    def __init__(self, backend: Any):
        """Initialize provenance wrapper.

        Args:
            backend: HybridQuantumBackend instance to wrap
        """
        self.backend = backend
        self._records: dict[str, ProvenanceRecord] = {}
        self._execution_chain: list[str] = []

    def execute_with_provenance(
        self,
        circuit: Any,
        shots: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, ProvenanceRecord]:
        """Execute circuit with full provenance tracking.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            metadata: Additional metadata to include

        Returns:
            Tuple of (execution_result, provenance_record)
        """
        record_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())

        # Compute input hash
        input_hash = self._compute_input_hash(circuit, shots)

        # Get circuit metrics
        circuit_depth = circuit.depth() if hasattr(circuit, "depth") else 0
        circuit_gates = len(circuit) if hasattr(circuit, "__len__") else 0

        # Get backend info
        backend_info = self.backend.get_backend_info()

        # Create initial provenance record
        record = ProvenanceRecord(
            record_id=record_id,
            execution_id=execution_id,
            input_hash=input_hash,
            backend_provider=backend_info.get("provider", "unknown"),
            backend_device=backend_info.get("backend_name", "unknown"),
            circuit_depth=circuit_depth,
            circuit_gates=circuit_gates,
            shots=shots or getattr(self.backend.config, "shots", 1024),
            metadata=metadata or {},
            parent_record_id=self._execution_chain[-1] if self._execution_chain else None,
        )

        # Execute circuit
        start_time = time.time()
        result = self.backend.execute_circuit(circuit, shots=shots)
        execution_time = time.time() - start_time

        # Update record with output hash
        record.output_hash = self._compute_output_hash(result)
        record.status = ProvenanceStatus.RECORDED
        record.metadata["execution_time"] = execution_time

        # Store record
        self._records[record_id] = record
        self._execution_chain.append(record_id)

        return result, record

    def verify_provenance(self, record_id: str, result: Any) -> bool:
        """Verify that result matches recorded provenance.

        Args:
            record_id: ID of provenance record to verify
            result: Execution result to verify

        Returns:
            True if result matches provenance record
        """
        if record_id not in self._records:
            return False

        record = self._records[record_id]
        computed_hash = self._compute_output_hash(result)

        if computed_hash == record.output_hash:
            record.status = ProvenanceStatus.VERIFIED
            return True
        else:
            record.status = ProvenanceStatus.REJECTED
            return False

    def get_record(self, record_id: str) -> ProvenanceRecord | None:
        """Get provenance record by ID.

        Args:
            record_id: Record ID to look up

        Returns:
            ProvenanceRecord or None if not found
        """
        return self._records.get(record_id)

    def get_execution_chain(self) -> list[ProvenanceRecord]:
        """Get full execution chain with provenance.

        Returns:
            List of ProvenanceRecord objects in execution order
        """
        return [self._records[rid] for rid in self._execution_chain if rid in self._records]

    def export_provenance_log(self) -> str:
        """Export provenance log as JSON.

        Returns:
            JSON string of all provenance records
        """
        records = [r.to_dict() for r in self._records.values()]
        return json.dumps({"records": records, "chain": self._execution_chain}, indent=2)

    def _compute_input_hash(self, circuit: Any, shots: int | None) -> str:
        """Compute deterministic hash of inputs."""
        circuit_str = str(circuit) if circuit is not None else ""
        shots_str = str(shots or 0)
        combined = f"{circuit_str}|{shots_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_output_hash(self, result: Any) -> str:
        """Compute hash of execution result."""
        if hasattr(result, "counts"):
            counts = result.counts
        elif isinstance(result, dict) and "counts" in result:
            counts = result["counts"]
        else:
            counts = str(result)

        counts_str = json.dumps(counts, sort_keys=True) if isinstance(counts, dict) else str(counts)
        return hashlib.sha256(counts_str.encode()).hexdigest()

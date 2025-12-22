"""Base Substrate Adapter - Zero Policy, Contract-Only.

This module implements the base adapter interface for substrate adapters.
CRITICAL: Adapters contain ZERO policy logic and accept ONLY valid contracts.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qcore.issuer import ContractBundle


class AdapterError(Exception):
    """Exception raised for adapter failures (FATAL)."""

    pass


@dataclass
class ExecutionProof:
    """Cryptographic proof of execution.

    Attributes:
        proof_id: Unique proof identifier (hash)
        contract_id: Reference to executed contract
        cluster_type: Cluster type used
        timestamp: Execution timestamp
        execution_hash: Hash of execution result
        deterministic: Whether execution was deterministic
        metadata: Additional metadata
    """

    proof_id: str
    contract_id: str
    cluster_type: str
    timestamp: str
    execution_hash: str
    deterministic: bool
    metadata: dict[str, Any]


class BaseAdapter(ABC):
    """Base adapter for substrate cluster execution.

    CRITICAL CONSTRAINTS:
    1. Adapters accept ONLY valid contracts from QRATUM
    2. Adapters contain ZERO policy logic
    3. All executions emit deterministic, hash-chained events
    4. Adapters MUST return ExecutionProof with cryptographic verification
    5. Adapters raise AdapterError (FATAL) on any violation
    """

    def __init__(self, cluster_type: str) -> None:
        """Initialize base adapter.

        Args:
            cluster_type: Type of cluster this adapter manages
        """
        self.cluster_type = cluster_type
        self._execution_count = 0

    @abstractmethod
    def execute(self, contract_bundle: ContractBundle) -> Any:
        """Execute a contract bundle.

        CRITICAL: This method ONLY executes with a valid contract.
        It NEVER makes policy decisions.

        Args:
            contract_bundle: Validated contract bundle

        Returns:
            Execution result (adapter-specific)

        Raises:
            AdapterError: If execution fails or contract is invalid
        """
        pass

    def validate_contract(self, contract_bundle: ContractBundle) -> None:
        """Validate contract bundle for this adapter.

        Args:
            contract_bundle: Contract bundle to validate

        Raises:
            AdapterError: If contract is invalid (FATAL)
        """
        # Verify contract references this cluster type
        expected_cluster = contract_bundle.capability_contract.get_cluster_type()
        if expected_cluster != self.cluster_type:
            raise AdapterError(
                f"Contract cluster type '{expected_cluster}' does not match "
                f"adapter cluster type '{self.cluster_type}'"
            )

        # Verify all contracts reference the same intent
        intent_id = contract_bundle.intent_contract.contract_id
        if contract_bundle.capability_contract.intent_contract_id != intent_id:
            raise AdapterError("Capability contract intent_id mismatch")
        if contract_bundle.temporal_contract.intent_contract_id != intent_id:
            raise AdapterError("Temporal contract intent_id mismatch")
        if contract_bundle.event_contract.intent_contract_id != intent_id:
            raise AdapterError("Event contract intent_id mismatch")

        # Verify temporal contract not expired
        if contract_bundle.temporal_contract.is_expired():
            raise AdapterError("Temporal contract has expired (FATAL)")

    def create_execution_proof(
        self,
        contract_id: str,
        execution_result: Any,
        deterministic: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionProof:
        """Create execution proof with cryptographic verification.

        Args:
            contract_id: Contract identifier
            execution_result: Result of execution
            deterministic: Whether execution was deterministic
            metadata: Optional additional metadata

        Returns:
            ExecutionProof with cryptographic verification
        """
        from datetime import timezone
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Compute execution hash
        result_str = str(execution_result)
        execution_hash = hashlib.sha256(result_str.encode("utf-8")).hexdigest()

        # Create proof ID
        proof_content = f"{contract_id}:{self.cluster_type}:{timestamp}:{execution_hash}"
        proof_id = hashlib.sha256(proof_content.encode("utf-8")).hexdigest()

        return ExecutionProof(
            proof_id=proof_id,
            contract_id=contract_id,
            cluster_type=self.cluster_type,
            timestamp=timestamp,
            execution_hash=execution_hash,
            deterministic=deterministic,
            metadata=metadata or {},
        )

    def log_execution_event(
        self, contract_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Log execution event (adapter-specific).

        Args:
            contract_id: Contract identifier
            event_type: Type of event
            payload: Event payload
        """
        # Import here to avoid circular dependency
        from events import log_event

        log_event(
            event_type=event_type,
            contract_id=contract_id,
            payload={
                **payload,
                "cluster_type": self.cluster_type,
                "adapter": self.__class__.__name__,
            },
        )

    def increment_execution_count(self) -> int:
        """Increment and return execution count.

        Returns:
            Current execution count
        """
        self._execution_count += 1
        return self._execution_count

    def get_execution_count(self) -> int:
        """Get total execution count.

        Returns:
            Total executions performed by this adapter
        """
        return self._execution_count

    def get_cluster_info(self) -> dict[str, Any]:
        """Get cluster information.

        Returns:
            Dictionary with cluster info
        """
        return {
            "cluster_type": self.cluster_type,
            "adapter_class": self.__class__.__name__,
            "execution_count": self._execution_count,
        }

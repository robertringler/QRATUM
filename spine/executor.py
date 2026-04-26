"""Deterministic Execution Spine - Contract Executor.

This module implements the contract executor, which dispatches validated
contracts to substrate adapters for execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from events import log_event
from qcore.issuer import ContractBundle


@dataclass
class ExecutionResult:
    """Result of contract execution.

    Attributes:
        success: Whether execution succeeded
        contract_id: ID of executed contract
        cluster_type: Cluster type used
        execution_time_seconds: Execution time
        proof: Execution proof
        metadata: Additional metadata
    """

    success: bool
    contract_id: str
    cluster_type: str
    execution_time_seconds: float
    proof: str
    metadata: dict[str, Any]


class ContractExecutor:
    """Executes contracts by dispatching to adapters."""

    def __init__(self, adapter_registry: Any = None) -> None:
        """Initialize contract executor.

        Args:
            adapter_registry: Optional adapter registry
        """
        self.adapter_registry = adapter_registry
        self._execution_history: dict[str, ExecutionResult] = {}

    def execute(self, contract_bundle: ContractBundle) -> ExecutionResult:
        """Execute a contract bundle.

        Args:
            contract_bundle: Bundle of contracts to execute

        Returns:
            ExecutionResult

        Raises:
            ValueError: If execution fails
        """
        intent_contract = contract_bundle.intent_contract
        capability_contract = contract_bundle.capability_contract
        temporal_contract = contract_bundle.temporal_contract

        # Log execution start
        log_event(
            event_type="ExecutionStarted",
            contract_id=intent_contract.contract_id,
            payload={
                "intent_name": intent_contract.intent_name,
                "cluster_type": capability_contract.get_cluster_type(),
            },
        )

        # Verify temporal contract
        if temporal_contract.is_expired():
            log_event(
                event_type="ExecutionFailed",
                contract_id=intent_contract.contract_id,
                payload={"reason": "Temporal contract expired"},
            )
            raise ValueError("Temporal contract has expired")

        # Get adapter for cluster type
        cluster_type = capability_contract.get_cluster_type()

        if self.adapter_registry:
            adapter = self.adapter_registry.get_adapter(cluster_type)
            if not adapter:
                raise ValueError(f"No adapter found for cluster type: {cluster_type}")

            # Dispatch to adapter
            execution_result = adapter.execute(contract_bundle)
        else:
            # Simulation mode (no adapter)
            execution_result = ExecutionResult(
                success=True,
                contract_id=intent_contract.contract_id,
                cluster_type=cluster_type,
                execution_time_seconds=0.0,
                proof=f"EXEC_{intent_contract.intent_name}_SIMULATED",
                metadata={"mode": "simulation"},
            )

        # Log execution completion
        log_event(
            event_type="ExecutionCompleted",
            contract_id=intent_contract.contract_id,
            payload={
                "success": execution_result.success,
                "execution_time_seconds": execution_result.execution_time_seconds,
                "proof": execution_result.proof,
            },
        )

        # Log audit
        log_event(
            event_type="AuditLogged",
            contract_id=intent_contract.contract_id,
            payload={
                "intent_name": intent_contract.intent_name,
                "cluster_type": cluster_type,
                "execution_result": execution_result.success,
            },
        )

        # Store execution history
        self._execution_history[intent_contract.contract_id] = execution_result

        return execution_result

    def get_execution_result(self, contract_id: str) -> ExecutionResult | None:
        """Get execution result for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            ExecutionResult if found, None otherwise
        """
        return self._execution_history.get(contract_id)

    def validate_contract_bundle(self, contract_bundle: ContractBundle) -> bool:
        """Validate a contract bundle before execution.

        Args:
            contract_bundle: Bundle to validate

        Returns:
            True if valid, False otherwise
        """
        # Verify all contracts reference the same intent
        intent_id = contract_bundle.intent_contract.contract_id

        if contract_bundle.capability_contract.intent_contract_id != intent_id:
            return False
        if contract_bundle.temporal_contract.intent_contract_id != intent_id:
            return False
        if contract_bundle.event_contract.intent_contract_id != intent_id:
            return False

        # Verify temporal contract is not expired
        return not contract_bundle.temporal_contract.is_expired()

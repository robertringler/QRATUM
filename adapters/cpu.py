"""High-Core-Count CPU Adapter.

This adapter interfaces with high-core-count CPU systems.
Contains ZERO policy logic - accepts ONLY valid contracts.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from adapters.base import AdapterError, BaseAdapter
from qcore.issuer import ContractBundle
from spine.executor import ExecutionResult


class CPUAdapter(BaseAdapter):
    """Adapter for high-core-count CPU systems."""

    def __init__(self) -> None:
        """Initialize CPU adapter."""
        super().__init__(cluster_type="CPU")

    def execute(self, contract_bundle: ContractBundle) -> ExecutionResult:
        """Execute contract on CPU.

        Args:
            contract_bundle: Validated contract bundle

        Returns:
            ExecutionResult

        Raises:
            AdapterError: If execution fails (FATAL)
        """
        # Validate contract (FATAL if invalid)
        self.validate_contract(contract_bundle)

        intent_contract = contract_bundle.intent_contract
        capability_contract = contract_bundle.capability_contract

        self.log_execution_event(
            contract_id=intent_contract.contract_id,
            event_type="CPUExecutionStarted",
            payload={"intent_name": intent_contract.intent_name},
        )

        try:
            # Get resource requirements
            topology = capability_contract.cluster_topology
            total_cores = topology.get("node_count", 1) * topology.get(
                "accelerators_per_node", 64
            )

            # Simulate CPU execution
            # In production, this would use standard CPU APIs
            execution_data = {
                "cluster_type": "CPU",
                "cores": total_cores,
                "memory_gb": topology.get("memory_per_node_gb", 512),
                "interconnect": "PCIe",
                "intent": intent_contract.intent_name,
            }

            # Create execution proof
            proof = self.create_execution_proof(
                contract_id=intent_contract.contract_id,
                execution_result=execution_data,
                deterministic=True,
                metadata={"architecture": "x86_64"},
            )

            self.increment_execution_count()

            self.log_execution_event(
                contract_id=intent_contract.contract_id,
                event_type="CPUExecutionCompleted",
                payload={"proof_id": proof.proof_id},
            )

            return ExecutionResult(
                success=True,
                contract_id=intent_contract.contract_id,
                cluster_type=self.cluster_type,
                execution_time_seconds=0.2,
                proof=proof.proof_id,
                metadata={"execution_data": execution_data},
            )

        except Exception as e:
            self.log_execution_event(
                contract_id=intent_contract.contract_id,
                event_type="CPUExecutionFailed",
                payload={"error": str(e)},
            )
            raise AdapterError(f"CPU execution failed: {e}") from e

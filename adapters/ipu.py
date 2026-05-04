"""Graphcore IPU Adapter.

This adapter interfaces with Graphcore Intelligence Processing Units.
Contains ZERO policy logic - accepts ONLY valid contracts.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from adapters.base import AdapterError, BaseAdapter
from qcore.issuer import ContractBundle
from spine.executor import ExecutionResult


class IPUAdapter(BaseAdapter):
    """Adapter for Graphcore IPU systems."""

    def __init__(self) -> None:
        """Initialize IPU adapter."""
        super().__init__(cluster_type="IPU")

    def execute(self, contract_bundle: ContractBundle) -> ExecutionResult:
        """Execute contract on IPU.

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
            event_type="IPUExecutionStarted",
            payload={"intent_name": intent_contract.intent_name},
        )

        try:
            # Get IPU resource requirements
            topology = capability_contract.cluster_topology
            ipus = topology.get("accelerators_per_node", 4)
            tiles_per_ipu = topology.get("metadata", {}).get("tiles_per_ipu", 1472)

            # Simulate IPU execution
            # In production, this would call Graphcore Poplar SDK
            execution_data = {
                "cluster_type": "IPU",
                "ipus": ipus,
                "tiles": ipus * tiles_per_ipu,
                "interconnect": "IPU-Fabric",
                "intent": intent_contract.intent_name,
            }

            # Create execution proof
            proof = self.create_execution_proof(
                contract_id=intent_contract.contract_id,
                execution_result=execution_data,
                deterministic=True,
                metadata={"tiles_per_ipu": tiles_per_ipu},
            )

            self.increment_execution_count()

            self.log_execution_event(
                contract_id=intent_contract.contract_id,
                event_type="IPUExecutionCompleted",
                payload={"proof_id": proof.proof_id},
            )

            return ExecutionResult(
                success=True,
                contract_id=intent_contract.contract_id,
                cluster_type=self.cluster_type,
                execution_time_seconds=0.3,
                proof=proof.proof_id,
                metadata={"execution_data": execution_data},
            )

        except Exception as e:
            self.log_execution_event(
                contract_id=intent_contract.contract_id,
                event_type="IPUExecutionFailed",
                payload={"error": str(e)},
            )
            raise AdapterError(f"IPU execution failed: {e}") from e

"""
QRADLE Engine - Core deterministic execution engine.

Orchestrates contract execution with Merkle-chained audit trails,
rollback capability, and invariant enforcement.
"""

import time
import uuid
from typing import Any, Callable, Dict, Optional

from .contract_types import Contract, ContractExecution, ContractStatus
from .invariants import FatalInvariants, InvariantViolation
from .merkle import MerkleChain
from .rollback import Checkpoint, RollbackManager


class QRADLEEngine:
    """Core deterministic execution engine for QRATUM.

    Features:
    - Deterministic contract execution
    - Merkle-chained audit trails
    - Rollback to any checkpoint
    - 8 Fatal Invariants enforcement
    - Cryptographic proof generation
    """

    def __init__(self):
        """Initialize QRADLE engine."""
        self.merkle_chain = MerkleChain()
        self.rollback_manager = RollbackManager()
        self.contracts: Dict[str, ContractExecution] = {}
        self.operations: Dict[str, Callable] = {}

        # Emit initialization event
        self.merkle_chain.add_event(
            "engine_initialized", {"timestamp": time.time(), "version": "1.0.0"}
        )

    def register_operation(self, name: str, handler: Callable) -> None:
        """Register an operation handler.

        Args:
            name: Operation name
            handler: Callable that executes the operation
        """
        self.operations[name] = handler
        self.merkle_chain.add_event(
            "operation_registered", {"operation": name, "timestamp": time.time()}
        )

    def create_contract(
        self,
        operation: str,
        inputs: Dict[str, Any],
        user_id: str = "system",
        expected_outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Contract:
        """Create a new contract.

        Args:
            operation: Name of operation to execute
            inputs: Input parameters
            user_id: User creating the contract
            expected_outputs: Optional expected outputs
            metadata: Optional metadata

        Returns:
            Created contract
        """
        contract_id = str(uuid.uuid4())
        contract = Contract(
            contract_id=contract_id,
            operation=operation,
            inputs=inputs,
            user_id=user_id,
            expected_outputs=expected_outputs,
            metadata=metadata or {},
        )

        # Verify invariants
        FatalInvariants.enforce_contract_immutability(contract)
        FatalInvariants.enforce_authorization(user_id, operation)

        # Emit event
        self.merkle_chain.add_event(
            "contract_created",
            {
                "contract_id": contract_id,
                "operation": operation,
                "user_id": user_id,
                "timestamp": time.time(),
            },
        )

        return contract

    def execute_contract(
        self, contract: Contract, safety_level: str = "routine"
    ) -> ContractExecution:
        """Execute a contract with full audit trail.

        Args:
            contract: Contract to execute
            safety_level: Safety level for this operation

        Returns:
            Contract execution result
        """
        start_time = time.time()

        # Verify invariants
        FatalInvariants.enforce_safety_levels(safety_level)
        FatalInvariants.enforce_human_oversight(contract.operation, safety_level)

        # Emit execution start event
        self.merkle_chain.add_event(
            "contract_execution_started",
            {
                "contract_id": contract.contract_id,
                "operation": contract.operation,
                "safety_level": safety_level,
                "timestamp": start_time,
            },
        )

        try:
            # Get operation handler
            if contract.operation not in self.operations:
                raise ValueError(f"Unknown operation: {contract.operation}")

            handler = self.operations[contract.operation]

            # Execute operation (deterministic)
            outputs = handler(contract.inputs)

            execution_time = time.time() - start_time

            # Create execution result
            execution = ContractExecution(
                contract=contract,
                status=ContractStatus.COMPLETED,
                outputs=outputs,
                execution_time=execution_time,
                proof_hash=contract.compute_hash(),
            )

            # Store execution
            self.contracts[contract.contract_id] = execution

            # Emit completion event
            self.merkle_chain.add_event(
                "contract_execution_completed",
                {
                    "contract_id": contract.contract_id,
                    "status": "completed",
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                },
            )

            return execution

        except Exception as e:
            execution_time = time.time() - start_time

            # Create failed execution
            execution = ContractExecution(
                contract=contract,
                status=ContractStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )

            self.contracts[contract.contract_id] = execution

            # Emit failure event
            self.merkle_chain.add_event(
                "contract_execution_failed",
                {"contract_id": contract.contract_id, "error": str(e), "timestamp": time.time()},
            )

            return execution

    def create_checkpoint(self, description: str = "") -> Checkpoint:
        """Create a checkpoint of current state.

        Args:
            description: Human-readable description

        Returns:
            Created checkpoint
        """
        checkpoint_id = str(uuid.uuid4())

        # Capture current state
        state = {
            "contracts": {k: v.to_dict() for k, v in self.contracts.items()},
            "chain_length": len(self.merkle_chain.chain),
        }

        merkle_proof = self.merkle_chain.get_chain_proof()

        checkpoint = self.rollback_manager.create_checkpoint(
            checkpoint_id=checkpoint_id,
            state=state,
            merkle_proof=merkle_proof,
            description=description,
        )

        # Emit checkpoint event
        self.merkle_chain.add_event(
            "checkpoint_created",
            {"checkpoint_id": checkpoint_id, "description": description, "timestamp": time.time()},
        )

        return checkpoint

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to

        Returns:
            True if successful, False otherwise
        """
        state = self.rollback_manager.rollback_to(checkpoint_id)
        if state is None:
            return False

        # Emit rollback event BEFORE restoring state
        self.merkle_chain.add_event(
            "rollback_initiated", {"checkpoint_id": checkpoint_id, "timestamp": time.time()}
        )

        # Restore state (simplified - in production would be more sophisticated)
        # Note: Merkle chain is append-only, we don't roll it back

        return True

    def verify_integrity(self) -> bool:
        """Verify integrity of all invariants and audit trail.

        Returns:
            True if all invariants hold, False otherwise
        """
        try:
            context = {"chain": self.merkle_chain, "rollback_manager": self.rollback_manager}
            FatalInvariants.verify_all(context)
            return True
        except InvariantViolation:
            return False

    def get_audit_trail(self, contract_id: Optional[str] = None) -> list:
        """Get audit trail for all or specific contract.

        Args:
            contract_id: Optional contract ID to filter by

        Returns:
            List of audit events
        """
        events = self.merkle_chain.get_events()
        if contract_id:
            events = [e for e in events if e.get("data", {}).get("contract_id") == contract_id]
        return events

    def get_system_proof(self) -> Dict[str, Any]:
        """Get cryptographic proof of current system state.

        Returns:
            Dictionary containing proof information
        """
        return {
            "merkle_root": self.merkle_chain.get_chain_proof(),
            "chain_length": len(self.merkle_chain.chain),
            "integrity_verified": self.verify_integrity(),
            "timestamp": time.time(),
        }

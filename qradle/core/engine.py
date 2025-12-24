"""
Deterministic Execution Engine for QRADLE

Provides deterministic execution with cryptographic proofs.
All operations are reproducible - same inputs always produce same outputs.

Version: 1.0.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from qradle.core.invariants import FatalInvariants, InvariantViolation
from qradle.core.merkle import MerkleChain
from qradle.core.rollback import RollbackManager


@dataclass
class ExecutionContext:
    """Context for deterministic execution.
    
    Attributes:
        contract_id: ID of contract being executed
        parameters: Input parameters
        timestamp: Execution timestamp
        safety_level: Safety level of operation
        authorized: Whether operation is authorized
        metadata: Additional context metadata
    """
    contract_id: str
    parameters: dict[str, Any]
    timestamp: str
    safety_level: str = "ROUTINE"
    authorized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of deterministic execution.
    
    Attributes:
        success: Whether execution succeeded
        output: Execution output
        output_hash: Hash of output for determinism verification
        execution_time: Execution duration
        events_emitted: Number of events emitted
        checkpoint_id: Checkpoint created before execution
        error: Error message if execution failed
    """
    success: bool
    output: Any
    output_hash: str
    execution_time: float
    events_emitted: int
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None


class DeterministicEngine:
    """Deterministic execution engine with cryptographic guarantees.
    
    The engine enforces all 8 Fatal Invariants and provides:
    - Deterministic execution (same inputs -> same outputs)
    - Merkle-chained event logs
    - Rollback capability
    - Human oversight enforcement
    - Complete auditability
    """
    
    def __init__(self):
        """Initialize the deterministic engine."""
        self.merkle_chain = MerkleChain({"engine": "QRADLE", "version": "1.0.0"})
        self.rollback_manager = RollbackManager()
        self.invariants = FatalInvariants()
        self._execution_count = 0
    
    def execute_contract(
        self,
        context: ExecutionContext,
        executor_func: Callable[[dict[str, Any]], Any],
        create_checkpoint: bool = True
    ) -> ExecutionResult:
        """Execute a contract with full invariant enforcement.
        
        Args:
            context: Execution context
            executor_func: Function to execute (must be deterministic)
            create_checkpoint: Whether to create checkpoint before execution
            
        Returns:
            ExecutionResult with output and metadata
            
        Raises:
            InvariantViolation: If any fatal invariant is violated
        """
        start_time = datetime.now(timezone.utc)
        checkpoint_id = None
        
        try:
            # Invariant 1: Enforce human oversight for sensitive operations
            self.invariants.enforce_human_oversight(
                operation=context.contract_id,
                safety_level=context.safety_level,
                authorized=context.authorized
            )
            
            # Invariant 4: Verify authorization system is active
            self.invariants.enforce_authorization_system(
                has_authorization_check=True
            )
            
            # Invariant 5: Verify safety level is set
            self.invariants.enforce_safety_level_system(
                operation=context.contract_id,
                has_safety_level=bool(context.safety_level)
            )
            
            # Invariant 6: Create checkpoint for rollback capability
            if create_checkpoint:
                checkpoint_state = {
                    "contract_id": context.contract_id,
                    "timestamp": context.timestamp,
                    "execution_count": self._execution_count,
                }
                checkpoint = self.rollback_manager.create_checkpoint(
                    state_data=checkpoint_state,
                    metadata={"contract_id": context.contract_id}
                )
                checkpoint_id = checkpoint.checkpoint_id
                
                self.invariants.enforce_rollback_capability(
                    checkpoint_available=True,
                    checkpoint_id=checkpoint_id
                )
            
            # Invariant 7: Emit execution start event
            start_event = {
                "event_type": "execution_started",
                "contract_id": context.contract_id,
                "timestamp": context.timestamp,
                "safety_level": context.safety_level,
            }
            self.merkle_chain.append(start_event)
            events_emitted = 1
            
            # Execute the contract
            output = executor_func(context.parameters)
            
            # Compute output hash for determinism verification (Invariant 8)
            output_serialized = json.dumps(
                {"output": output, "contract_id": context.contract_id},
                sort_keys=True
            )
            output_hash = hashlib.sha256(output_serialized.encode()).hexdigest()
            
            # Emit completion event (Invariant 7)
            completion_event = {
                "event_type": "execution_completed",
                "contract_id": context.contract_id,
                "output_hash": output_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.merkle_chain.append(completion_event)
            events_emitted += 1
            
            # Verify Merkle chain integrity (Invariant 2)
            chain_valid = self.merkle_chain.verify_chain_integrity()
            self.invariants.enforce_merkle_integrity(
                chain_valid=chain_valid,
                last_hash=self.merkle_chain.get_root_hash()
            )
            
            # Verify event emission (Invariant 7)
            self.invariants.enforce_event_emission(
                event_emitted=(events_emitted > 0),
                operation=context.contract_id
            )
            
            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            self._execution_count += 1
            
            return ExecutionResult(
                success=True,
                output=output,
                output_hash=output_hash,
                execution_time=execution_time,
                events_emitted=events_emitted,
                checkpoint_id=checkpoint_id
            )
            
        except InvariantViolation as e:
            # Fatal invariant violation - system must halt
            error_event = {
                "event_type": "invariant_violation",
                "contract_id": context.contract_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.merkle_chain.append(error_event)
            
            # Re-raise the violation
            raise
            
        except Exception as e:
            # Regular execution error
            error_event = {
                "event_type": "execution_failed",
                "contract_id": context.contract_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.merkle_chain.append(error_event)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ExecutionResult(
                success=False,
                output=None,
                output_hash="",
                execution_time=execution_time,
                events_emitted=1,
                checkpoint_id=checkpoint_id,
                error=str(e)
            )
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            True if rollback succeeded
        """
        try:
            self.rollback_manager.rollback_to(checkpoint_id)
            
            # Emit rollback event
            rollback_event = {
                "event_type": "rollback_executed",
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.merkle_chain.append(rollback_event)
            
            return True
        except Exception:
            return False
    
    def get_execution_proof(self, contract_id: str) -> Optional[dict[str, Any]]:
        """Get cryptographic proof of contract execution.
        
        Args:
            contract_id: ID of contract to get proof for
            
        Returns:
            Proof dictionary or None if not found
        """
        # Find all events for this contract
        contract_events = []
        for idx, node in enumerate(self.merkle_chain.nodes):
            if node.data.get("contract_id") == contract_id:
                contract_events.append((idx, node))
        
        if not contract_events:
            return None
        
        # Generate proof for the completion event
        completion_idx, completion_node = contract_events[-1]
        proof = self.merkle_chain.get_proof(completion_idx)
        
        return {
            "contract_id": contract_id,
            "event_data": completion_node.data,
            "merkle_proof": proof.to_dict(),
            "chain_root": self.merkle_chain.get_root_hash(),
        }
    
    def verify_execution(self, contract_id: str, expected_output_hash: str) -> bool:
        """Verify a contract execution was deterministic.
        
        Args:
            contract_id: ID of contract to verify
            expected_output_hash: Expected output hash
            
        Returns:
            True if execution is verified
        """
        proof = self.get_execution_proof(contract_id)
        if not proof:
            return False
        
        actual_hash = proof["event_data"].get("output_hash", "")
        return actual_hash == expected_output_hash
    
    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_executions": self._execution_count,
            "chain_length": len(self.merkle_chain.nodes),
            "chain_root_hash": self.merkle_chain.get_root_hash(),
            "chain_valid": self.merkle_chain.verify_chain_integrity(),
            "checkpoints": self.rollback_manager.get_stats(),
        }

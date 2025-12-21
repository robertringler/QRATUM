"""Vertical Module Base Class.

Abstract base class enforcing all 8 fatal invariants for vertical modules.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional

from platform.core.events import EventType, ExecutionEvent, MerkleEventChain
from platform.core.intent import PlatformContract, PlatformIntent
from platform.core.substrates import ComputeSubstrate, select_optimal_substrate


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a vertical module execution.

    Attributes:
        success: Whether execution succeeded
        result_data: The actual result data
        warnings: Any warnings generated
        safety_disclaimer: Required safety disclaimer
        event_chain_root: Merkle root of event chain
        execution_time_ms: Execution time in milliseconds
        substrate_used: Compute substrate that was used
    """

    success: bool
    result_data: Dict[str, Any]
    warnings: List[str]
    safety_disclaimer: str
    event_chain_root: str
    execution_time_ms: float
    substrate_used: ComputeSubstrate


class VerticalModuleBase(ABC):
    """Abstract base class for all vertical modules.

    Enforces the 8 FATAL INVARIANTS:
    1. All execution must be deterministic
    2. Every operation must validate contract before execution
    3. Every step must emit events to Merkle chain
    4. Safety disclaimers must be present in all outputs
    5. Prohibited uses must be explicitly checked
    6. Compliance attestations must be validated
    7. Optimal substrate selection must occur
    8. Results must include complete audit trail

    Attributes:
        vertical_name: Name of the vertical module
        event_chain: Merkle event chain for audit
        seed: Random seed for determinism
    """

    def __init__(self, vertical_name: str, seed: int = 42):
        """Initialize vertical module.

        Args:
            vertical_name: Name of this vertical
            seed: Random seed for deterministic execution
        """
        self.vertical_name = vertical_name
        self.event_chain = MerkleEventChain()
        self.seed = seed
        self._execution_count = 0

    @abstractmethod
    def get_safety_disclaimer(self) -> str:
        """Get the safety disclaimer for this vertical.

        Returns:
            Safety disclaimer text

        FATAL INVARIANT #4: Must be implemented by all verticals.
        """
        pass

    @abstractmethod
    def get_prohibited_uses(self) -> FrozenSet[str]:
        """Get the set of prohibited uses for this vertical.

        Returns:
            Set of prohibited use descriptions

        FATAL INVARIANT #5: Must be implemented by all verticals.
        """
        pass

    @abstractmethod
    def get_required_attestations(self, operation: str) -> FrozenSet[str]:
        """Get required compliance attestations for an operation.

        Args:
            operation: The operation being performed

        Returns:
            Set of required attestation identifiers

        FATAL INVARIANT #6: Must be implemented by all verticals.
        """
        pass

    @abstractmethod
    def _execute_operation(
        self, contract: PlatformContract, substrate: ComputeSubstrate
    ) -> Dict[str, Any]:
        """Execute the actual operation (internal implementation).

        Args:
            contract: Validated execution contract
            substrate: Selected compute substrate

        Returns:
            Operation result data

        FATAL INVARIANT #1: Implementation must be deterministic.
        FATAL INVARIANT #3: Implementation must emit events.
        """
        pass

    def _validate_contract(self, contract: PlatformContract) -> None:
        """Validate execution contract.

        Args:
            contract: Contract to validate

        Raises:
            ValueError: If contract is invalid

        FATAL INVARIANT #2: Contract must be validated before execution.
        """
        # Emit validation event
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.VALIDATION_CHECK,
                vertical=self.vertical_name,
                operation=contract.intent.operation,
                payload={"validation": "contract_check", "contract_hash": contract.contract_hash},
            )
        )

        # Check contract validity
        if not contract.is_valid():
            raise ValueError(f"Invalid contract: not authorized or expired")

        # Check required attestations
        required = self.get_required_attestations(contract.intent.operation)
        for attestation in required:
            if not contract.has_attestation(attestation):
                raise ValueError(f"Missing required attestation: {attestation}")

        # Check for prohibited uses
        prohibited = self.get_prohibited_uses()
        operation_lower = contract.intent.operation.lower()
        for prohibited_use in prohibited:
            if prohibited_use.lower() in operation_lower:
                raise ValueError(f"Prohibited use detected: {prohibited_use}")

    def _select_substrate(
        self, operation: str, parameters: Dict[str, Any]
    ) -> ComputeSubstrate:
        """Select optimal compute substrate.

        Args:
            operation: Operation to be performed
            parameters: Operation parameters

        Returns:
            Selected compute substrate

        FATAL INVARIANT #7: Optimal substrate must be selected.
        """
        # Extract problem characteristics
        problem_size = parameters.get("problem_size", 1000)
        task_type = parameters.get("task_type", operation)

        substrate = select_optimal_substrate(
            problem_size=problem_size,
            task_type=task_type,
            required_availability=0.8,
        )

        # Emit substrate selection event
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation=operation,
                payload={
                    "step": "substrate_selection",
                    "substrate": substrate.value,
                    "problem_size": problem_size,
                },
            )
        )

        return substrate

    def execute(self, contract: PlatformContract) -> ExecutionResult:
        """Execute operation with full invariant enforcement.

        Args:
            contract: Validated execution contract

        Returns:
            Execution result with audit trail

        This method enforces ALL 8 FATAL INVARIANTS.
        """
        start_time = datetime.now(timezone.utc)

        # FATAL INVARIANT #3: Emit execution start event
        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.EXECUTION_STARTED,
                vertical=self.vertical_name,
                operation=contract.intent.operation,
                payload={
                    "user_id": contract.intent.user_id,
                    "session_id": contract.intent.session_id,
                    "contract_hash": contract.contract_hash,
                },
            )
        )

        try:
            # FATAL INVARIANT #2: Validate contract
            self._validate_contract(contract)

            # FATAL INVARIANT #7: Select optimal substrate
            substrate = self._select_substrate(contract.intent.operation, contract.intent.parameters)

            # FATAL INVARIANT #3: Emit safety check event
            self.event_chain.append(
                ExecutionEvent(
                    event_type=EventType.SAFETY_CHECK,
                    vertical=self.vertical_name,
                    operation=contract.intent.operation,
                    payload={"safety_checks": "passed", "prohibited_uses": "checked"},
                )
            )

            # FATAL INVARIANT #1 & #3: Execute operation (must be deterministic and emit events)
            result_data = self._execute_operation(contract, substrate)

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # FATAL INVARIANT #3: Emit completion event
            self.event_chain.append(
                ExecutionEvent(
                    event_type=EventType.EXECUTION_COMPLETED,
                    vertical=self.vertical_name,
                    operation=contract.intent.operation,
                    payload={
                        "success": True,
                        "execution_time_ms": execution_time_ms,
                        "substrate": substrate.value,
                    },
                )
            )

            # FATAL INVARIANT #8: Return result with complete audit trail
            return ExecutionResult(
                success=True,
                result_data=result_data,
                warnings=[],
                safety_disclaimer=self.get_safety_disclaimer(),  # FATAL INVARIANT #4
                event_chain_root=self.event_chain.get_merkle_root(),
                execution_time_ms=execution_time_ms,
                substrate_used=substrate,
            )

        except Exception as e:
            # FATAL INVARIANT #3: Emit failure event
            self.event_chain.append(
                ExecutionEvent(
                    event_type=EventType.EXECUTION_FAILED,
                    vertical=self.vertical_name,
                    operation=contract.intent.operation,
                    payload={"error": str(e), "error_type": type(e).__name__},
                )
            )

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # FATAL INVARIANT #8: Return failure result with audit trail
            return ExecutionResult(
                success=False,
                result_data={"error": str(e)},
                warnings=[str(e)],
                safety_disclaimer=self.get_safety_disclaimer(),  # FATAL INVARIANT #4
                event_chain_root=self.event_chain.get_merkle_root(),
                execution_time_ms=execution_time_ms,
                substrate_used=ComputeSubstrate.CPU_SERIAL,
            )

    def get_event_chain(self) -> MerkleEventChain:
        """Get the event chain for audit purposes.

        Returns:
            The Merkle event chain
        """
        return self.event_chain

    def verify_audit_trail(self) -> bool:
        """Verify integrity of audit trail.

        Returns:
            True if audit trail is valid
        """
        return self.event_chain.verify_chain()

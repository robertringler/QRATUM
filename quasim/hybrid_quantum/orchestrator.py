"""Hybrid Quantum-Classical Orchestrator for QRATUM.

This module implements the bounded, invariant-preserving orchestrator for
managing quantum-classical hybrid workflows within QRATUM's trust-conserving
architecture.

Core Principles:
- Trust conservation: ℛ(t) ≥ 0
- Proposals only — never auto-commit
- Full rollback, determinism, provenance
- Quantum outputs verified before use
- Fallback to classical tensor on quantum failure

Architecture:
- Non-agentic: All operations externally triggered
- Proposal-only: No internal goal generation
- Deterministic: Reproducible execution paths
- Observable: Full audit trail
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np


# Trust adjustment constants for maintainability
TRUST_SUCCESS_MAX_INCREASE = 1.05  # Maximum trust increase factor on success
TRUST_SUCCESS_INCREMENT = 0.01  # Trust increment per unit below maximum
TRUST_FALLBACK_REDUCTION = 0.95  # Trust factor for fallback success
TRUST_FAILURE_REDUCTION_RATE = 0.1  # Trust reduction rate per retry on failure


class OrchestratorStatus(Enum):
    """Status of orchestrator execution."""

    IDLE = "idle"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    FALLBACK_ACTIVE = "fallback_active"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ExecutionMode(Enum):
    """Execution mode for hybrid operations."""

    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class FailureType(Enum):
    """Types of quantum execution failures."""

    DECOHERENCE = "decoherence"
    NOISE_THRESHOLD_EXCEEDED = "noise_threshold_exceeded"
    ZK_PROOF_FAILURE = "zk_proof_failure"
    TRANSIENT = "transient"
    HARDWARE_ERROR = "hardware_error"
    TIMEOUT = "timeout"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class TrustMetric:
    """Trust metric for QRATUM invariant enforcement.

    Implements ℛ(t) ≥ 0 invariant where:
    - ℛ(t) represents trust score at time t
    - Must always be non-negative
    - Tracks variance for P1 epistemic perfection target

    Attributes:
        value: Current trust score (must be ≥ 0)
        timestamp: When metric was computed
        variance: Trust variance (target ≤ 0.001)
        contributing_factors: Factors affecting trust
        history: Recent trust values for trend analysis
    """

    value: float = 1.0
    timestamp: str = ""
    variance: float = 0.0
    contributing_factors: dict[str, float] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate trust invariant."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        # Enforce ℛ(t) ≥ 0 invariant
        if self.value < 0:
            raise ValueError(f"Trust invariant violated: ℛ(t) = {self.value} < 0")

    def update(self, new_value: float, factors: dict[str, float] | None = None) -> None:
        """Update trust metric while preserving invariant.

        Args:
            new_value: New trust value
            factors: Contributing factors to this update

        Raises:
            ValueError: If new_value would violate invariant
        """
        if new_value < 0:
            raise ValueError(f"Trust invariant would be violated: ℛ(t) = {new_value} < 0")

        self.history.append(self.value)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        self.value = new_value
        self.timestamp = datetime.utcnow().isoformat()

        if factors:
            self.contributing_factors.update(factors)

        # Compute variance from history
        if len(self.history) >= 2:
            self.variance = float(np.var(self.history))

    @property
    def is_valid(self) -> bool:
        """Check if trust metric satisfies invariant."""
        return self.value >= 0

    @property
    def meets_p1_target(self) -> bool:
        """Check if variance meets P1 target (≤ 0.001)."""
        return self.variance <= 0.001


@dataclass
class FallbackStrategy:
    """Strategy for handling quantum execution failures.

    Attributes:
        max_retries: Maximum retry attempts (bounded at 3)
        retry_delay_seconds: Delay between retries
        fallback_to_classical: Whether to fallback to classical on failure
        classical_fallback_function: Optional classical fallback implementation
        escalate_on_failure: Whether to escalate to human on exhausted retries
    """

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    fallback_to_classical: bool = True
    classical_fallback_function: Callable[..., Any] | None = None
    escalate_on_failure: bool = True

    def __post_init__(self) -> None:
        """Enforce bounded retry limit."""
        if self.max_retries > 3:
            self.max_retries = 3  # Hard cap per spec


@dataclass
class ExecutionContext:
    """Context for a hybrid quantum-classical execution.

    Attributes:
        execution_id: Unique identifier for this execution
        mode: Current execution mode (quantum/classical/hybrid/fallback)
        status: Current status
        start_time: When execution started
        retry_count: Number of retries attempted
        failures: List of failures encountered
        provenance_chain: List of provenance hashes
        trust_metric: Current trust metric
        metadata: Additional context
    """

    execution_id: str
    mode: ExecutionMode = ExecutionMode.QUANTUM
    status: OrchestratorStatus = OrchestratorStatus.IDLE
    start_time: str = ""
    retry_count: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    provenance_chain: list[str] = field(default_factory=list)
    trust_metric: TrustMetric = field(default_factory=TrustMetric)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults."""
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat()

    def record_failure(self, failure_type: FailureType, details: str) -> None:
        """Record a failure in this execution context."""
        self.failures.append(
            {
                "type": failure_type.value,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_number": self.retry_count,
            }
        )

    def add_provenance(self, provenance_hash: str) -> None:
        """Add provenance hash to chain."""
        self.provenance_chain.append(provenance_hash)


class HybridQuantumOrchestrator:
    """Bounded, invariant-preserving orchestrator for quantum-classical workflows.

    This orchestrator coordinates quantum and classical computations while
    maintaining QRATUM's core invariants:
    - Trust conservation: ℛ(t) ≥ 0
    - No internal goal generation
    - All operations externally triggered
    - Proposals only — never auto-commit
    - Full rollback capability
    - Quantum outputs verified before use

    Example:
        >>> orchestrator = HybridQuantumOrchestrator()
        >>> context = orchestrator.create_execution_context()
        >>> result = orchestrator.execute_hybrid(
        ...     context=context,
        ...     quantum_op=quantum_circuit,
        ...     classical_fallback=classical_simulation,
        ... )
        >>> if result.requires_approval:
        ...     # Result is proposal-only, await human approval
        ...     pass
    """

    def __init__(
        self,
        fallback_strategy: FallbackStrategy | None = None,
        require_dual_approval: bool = True,
        zk_proof_latency_threshold: float = 5.0,
    ):
        """Initialize orchestrator.

        Args:
            fallback_strategy: Strategy for handling failures
            require_dual_approval: Whether dual-control is required
            zk_proof_latency_threshold: Max zk-proof latency in seconds (P0 target: ≤5s)
        """
        self.fallback_strategy = fallback_strategy or FallbackStrategy()
        self.require_dual_approval = require_dual_approval
        self.zk_proof_latency_threshold = zk_proof_latency_threshold

        self._active_contexts: dict[str, ExecutionContext] = {}
        self._execution_history: list[ExecutionContext] = []
        self._global_trust: TrustMetric = TrustMetric()
        self._pending_proposals: dict[str, dict[str, Any]] = {}

    def create_execution_context(
        self,
        mode: ExecutionMode = ExecutionMode.QUANTUM,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionContext:
        """Create a new execution context.

        Args:
            mode: Initial execution mode
            metadata: Additional context metadata

        Returns:
            New ExecutionContext with unique ID
        """
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            execution_id=execution_id,
            mode=mode,
            status=OrchestratorStatus.IDLE,
            metadata=metadata or {},
        )
        self._active_contexts[execution_id] = context
        return context

    def execute_hybrid(
        self,
        context: ExecutionContext,
        quantum_op: Callable[..., Any],
        classical_fallback: Callable[..., Any] | None = None,
        verification_fn: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute hybrid quantum-classical operation with fallback handling.

        This method implements the bounded retry logic:
        1. Attempt quantum execution
        2. On failure, retry up to max_retries (bounded at 3)
        3. On exhausted retries, fallback to classical
        4. On classical failure, escalate to human

        Args:
            context: Execution context
            quantum_op: Quantum operation to execute
            classical_fallback: Classical fallback function
            verification_fn: Optional verification function for output
            **kwargs: Arguments passed to quantum_op

        Returns:
            Result dictionary with execution outcome and metadata
        """
        context.status = OrchestratorStatus.EXECUTING
        result: dict[str, Any] = {
            "execution_id": context.execution_id,
            "success": False,
            "mode_used": context.mode.value,
            "requires_approval": self.require_dual_approval,
            "output": None,
            "provenance_hash": "",
            "trust_metric": None,
            "failures": [],
        }

        # Track fallback function
        fallback_fn = classical_fallback or self.fallback_strategy.classical_fallback_function

        # Quantum execution with bounded retry
        for attempt in range(self.fallback_strategy.max_retries + 1):
            context.retry_count = attempt
            try:
                # Execute quantum operation
                start_time = time.time()
                output = quantum_op(**kwargs)
                execution_time = time.time() - start_time

                # Compute provenance hash
                provenance_hash = self._compute_provenance_hash(kwargs, output)
                context.add_provenance(provenance_hash)
                result["provenance_hash"] = provenance_hash

                # Verify output if verification function provided
                if verification_fn is not None:
                    if not verification_fn(output):
                        raise QuantumVerificationError("Output failed verification")

                # Check zk-proof latency threshold
                if execution_time > self.zk_proof_latency_threshold:
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(
                        f"Execution time {execution_time:.2f}s exceeds P0 target of {self.zk_proof_latency_threshold}s"
                    )

                # Success
                result["success"] = True
                result["output"] = output
                result["mode_used"] = ExecutionMode.QUANTUM.value
                context.status = OrchestratorStatus.COMPLETED
                context.mode = ExecutionMode.QUANTUM

                # Update trust metric
                self._update_trust_metric(context, success=True)
                result["trust_metric"] = context.trust_metric.value

                break

            except Exception as e:
                failure_type = self._classify_failure(e)
                context.record_failure(failure_type, str(e))

                # Check if we should retry
                if attempt < self.fallback_strategy.max_retries:
                    time.sleep(self.fallback_strategy.retry_delay_seconds)
                    continue

                # Exhausted retries - try classical fallback
                if self.fallback_strategy.fallback_to_classical and fallback_fn is not None:
                    try:
                        output = fallback_fn(**kwargs)
                        result["success"] = True
                        result["output"] = output
                        result["mode_used"] = ExecutionMode.FALLBACK.value
                        context.status = OrchestratorStatus.FALLBACK_ACTIVE
                        context.mode = ExecutionMode.FALLBACK

                        provenance_hash = self._compute_provenance_hash(kwargs, output)
                        context.add_provenance(provenance_hash)
                        result["provenance_hash"] = provenance_hash

                        self._update_trust_metric(context, success=True, is_fallback=True)
                        result["trust_metric"] = context.trust_metric.value

                    except Exception as fallback_error:
                        context.record_failure(FailureType.HARDWARE_ERROR, str(fallback_error))
                        context.status = OrchestratorStatus.FAILED

                        if self.fallback_strategy.escalate_on_failure:
                            result["requires_human_escalation"] = True

                        self._update_trust_metric(context, success=False)
                        result["trust_metric"] = context.trust_metric.value
                else:
                    # No fallback available
                    context.status = OrchestratorStatus.FAILED
                    if self.fallback_strategy.escalate_on_failure:
                        result["requires_human_escalation"] = True

                    self._update_trust_metric(context, success=False)
                    result["trust_metric"] = context.trust_metric.value

        result["failures"] = context.failures
        result["retry_count"] = context.retry_count

        # Store in history
        self._execution_history.append(context)
        if context.execution_id in self._active_contexts:
            del self._active_contexts[context.execution_id]

        # If dual approval required, create proposal
        if self.require_dual_approval and result["success"]:
            proposal_id = self._create_proposal(context, result)
            result["proposal_id"] = proposal_id
            result["requires_approval"] = True
            context.status = OrchestratorStatus.AWAITING_APPROVAL

        return result

    def rollback_execution(self, execution_id: str, rollback_manager: Any) -> dict[str, Any]:
        """Rollback an execution to its pre-execution state.

        Args:
            execution_id: ID of execution to rollback
            rollback_manager: RollbackManager instance

        Returns:
            Rollback result dictionary
        """
        context = next(
            (c for c in self._execution_history if c.execution_id == execution_id), None
        )

        if context is None:
            return {
                "success": False,
                "error": f"Execution {execution_id} not found",
            }

        restored_state = rollback_manager.rollback_execution(execution_id)

        if restored_state is not None:
            context.status = OrchestratorStatus.ROLLED_BACK
            return {
                "success": True,
                "execution_id": execution_id,
                "restored_state": restored_state,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "success": False,
                "error": "No checkpoint found for execution",
            }

    def get_trust_report(self) -> dict[str, Any]:
        """Generate trust metric report with ℛ(t) ≥ 0 assertion.

        Returns:
            Trust report dictionary with audit trail
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "global_trust": {
                "value": self._global_trust.value,
                "variance": self._global_trust.variance,
                "is_valid": self._global_trust.is_valid,
                "meets_p1_target": self._global_trust.meets_p1_target,
                "history_length": len(self._global_trust.history),
            },
            "invariant_assertion": "ℛ(t) ≥ 0",
            "invariant_satisfied": self._global_trust.is_valid,
            "active_executions": len(self._active_contexts),
            "total_executions": len(self._execution_history),
            "pending_proposals": len(self._pending_proposals),
        }

    def get_pending_proposals(self) -> list[dict[str, Any]]:
        """Get all pending proposals awaiting approval.

        Returns:
            List of pending proposal dictionaries
        """
        return list(self._pending_proposals.values())

    def approve_proposal(
        self, proposal_id: str, approver_id: str, notes: str = ""
    ) -> dict[str, Any]:
        """Record approval for a proposal.

        Args:
            proposal_id: ID of proposal to approve
            approver_id: ID of approving entity
            notes: Optional approval notes

        Returns:
            Approval result dictionary
        """
        if proposal_id not in self._pending_proposals:
            return {"success": False, "error": "Proposal not found"}

        proposal = self._pending_proposals[proposal_id]
        proposal["approvals"].append(
            {
                "approver_id": approver_id,
                "timestamp": datetime.utcnow().isoformat(),
                "notes": notes,
            }
        )

        # Check if fully approved (dual control = 2 approvals)
        if len(proposal["approvals"]) >= 2:
            proposal["status"] = "approved"

        return {
            "success": True,
            "proposal_id": proposal_id,
            "approvals_count": len(proposal["approvals"]),
            "is_fully_approved": proposal["status"] == "approved",
        }

    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify exception into failure type."""
        error_str = str(error).lower()

        if "decoherence" in error_str:
            return FailureType.DECOHERENCE
        elif "noise" in error_str:
            return FailureType.NOISE_THRESHOLD_EXCEEDED
        elif "zk" in error_str or "proof" in error_str:
            return FailureType.ZK_PROOF_FAILURE
        elif "verification" in error_str:
            return FailureType.VERIFICATION_FAILED
        elif "timeout" in error_str:
            return FailureType.TIMEOUT
        elif "hardware" in error_str:
            return FailureType.HARDWARE_ERROR
        else:
            return FailureType.TRANSIENT

    def _compute_provenance_hash(
        self, inputs: dict[str, Any], outputs: Any
    ) -> str:
        """Compute deterministic provenance hash."""
        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        outputs_str = str(outputs)
        combined = f"{inputs_str}|{outputs_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _update_trust_metric(
        self,
        context: ExecutionContext,
        success: bool,
        is_fallback: bool = False,
    ) -> None:
        """Update trust metrics based on execution outcome."""
        # Compute trust adjustment using named constants
        if success:
            if is_fallback:
                # Fallback success: slight trust reduction
                adjustment = TRUST_FALLBACK_REDUCTION
            else:
                # Quantum success: trust preserved or increased
                adjustment = min(
                    TRUST_SUCCESS_MAX_INCREASE,
                    1.0 + TRUST_SUCCESS_INCREMENT * (1.0 - context.trust_metric.value)
                )
        else:
            # Failure: trust reduction proportional to retry count
            adjustment = max(
                0.0, 1.0 - TRUST_FAILURE_REDUCTION_RATE * (context.retry_count + 1)
            )

        new_value = max(0.0, context.trust_metric.value * adjustment)

        factors = {
            "success": float(success),
            "is_fallback": float(is_fallback),
            "retry_count": context.retry_count,
            "failure_count": len(context.failures),
        }

        context.trust_metric.update(new_value, factors)

        # Update global trust
        self._global_trust.update(
            new_value,
            {
                "execution_id": context.execution_id,
                **factors,
            },
        )

    def _create_proposal(
        self, context: ExecutionContext, result: dict[str, Any]
    ) -> str:
        """Create proposal artifact for approval workflow."""
        proposal_id = str(uuid.uuid4())

        proposal = {
            "proposal_id": proposal_id,
            "execution_id": context.execution_id,
            "cluster": self._determine_cluster(context),
            "metrics_target": {
                "primary": "execution_success",
                "secondary": "trust_preserved",
            },
            "inputs": context.provenance_chain[:1] if context.provenance_chain else [],
            "expected_outputs": {
                "artifacts": [result.get("provenance_hash", "")],
                "metrics_delta": {
                    "trust": result.get("trust_metric", 0),
                },
            },
            "fallback_strategy": self.fallback_strategy.max_retries,
            "rollback_path": context.provenance_chain[-1] if context.provenance_chain else "",
            "cryptographic_signatures": [],  # Placeholders for dual control
            "invariant_assertion": "ℛ(t) ≥ 0",
            "status": "pending",
            "approvals": [],
            "created_at": datetime.utcnow().isoformat(),
        }

        self._pending_proposals[proposal_id] = proposal
        return proposal_id

    def _determine_cluster(self, context: ExecutionContext) -> str:
        """Determine priority cluster for proposal."""
        if context.mode == ExecutionMode.QUANTUM:
            return "P0"  # Quantum-Classical Hybrid Speed
        elif len(context.failures) > 0:
            return "P3"  # Operational Anti-Entropy
        else:
            return "P1"  # Epistemic Perfection


class QuantumVerificationError(Exception):
    """Exception raised when quantum output fails verification."""

    pass

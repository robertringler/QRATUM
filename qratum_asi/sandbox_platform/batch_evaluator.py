"""Batch Proposal Evaluator for Simultaneous Evaluation.

Batches multiple proposals for simultaneous evaluation to improve
throughput and reduce overhead.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain
from qratum_asi.sandbox_platform.types import (
    ProposalPriority,
    SandboxEvaluationResult,
    SandboxProposal,
)


class BatchStatus(Enum):
    """Status of an evaluation batch."""

    COLLECTING = "collecting"
    READY = "ready"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvaluationBatch:
    """Batch of proposals for simultaneous evaluation.

    Attributes:
        batch_id: Unique batch identifier
        proposals: Proposals in the batch
        priority: Batch priority (highest of contained proposals)
        max_size: Maximum batch size
        timeout_ms: Batch collection timeout
        status: Current batch status
    """

    batch_id: str
    proposals: list[SandboxProposal] = field(default_factory=list)
    priority: ProposalPriority = ProposalPriority.NORMAL
    max_size: int = 50
    timeout_ms: int = 1000
    status: BatchStatus = BatchStatus.COLLECTING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    results: list[SandboxEvaluationResult] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        """Check if batch is full."""
        return len(self.proposals) >= self.max_size

    @property
    def is_ready(self) -> bool:
        """Check if batch is ready for evaluation."""
        return self.status == BatchStatus.READY or self.is_full

    def add_proposal(self, proposal: SandboxProposal) -> bool:
        """Add a proposal to the batch.

        Args:
            proposal: Proposal to add

        Returns:
            True if added successfully
        """
        if self.is_full or self.status != BatchStatus.COLLECTING:
            return False

        self.proposals.append(proposal)

        # Update priority to highest
        if proposal.priority.value < self.priority.value:
            self.priority = proposal.priority

        return True

    def compute_hash(self) -> str:
        """Compute hash of batch content."""
        content = {
            "batch_id": self.batch_id,
            "proposal_ids": [p.proposal_id for p in self.proposals],
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch."""
        return {
            "batch_id": self.batch_id,
            "proposal_count": len(self.proposals),
            "priority": self.priority.value,
            "max_size": self.max_size,
            "timeout_ms": self.timeout_ms,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_count": len(self.results),
            "batch_hash": self.compute_hash(),
        }


@dataclass
class BatchResult:
    """Result of batch evaluation.

    Attributes:
        batch_id: Source batch identifier
        success: Whether batch evaluation succeeded
        results: Individual proposal results
        total_time_ms: Total evaluation time
        avg_time_per_proposal_ms: Average time per proposal
        throughput: Proposals processed per second
    """

    batch_id: str
    success: bool
    results: list[SandboxEvaluationResult]
    total_time_ms: float = 0.0
    avg_time_per_proposal_ms: float = 0.0
    throughput: float = 0.0
    merkle_proof: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch result."""
        return {
            "batch_id": self.batch_id,
            "success": self.success,
            "result_count": len(self.results),
            "success_count": sum(1 for r in self.results if r.success),
            "total_time_ms": self.total_time_ms,
            "avg_time_per_proposal_ms": self.avg_time_per_proposal_ms,
            "throughput": self.throughput,
            "merkle_proof": self.merkle_proof,
            "created_at": self.created_at,
            "errors": self.errors,
        }


class BatchProposalEvaluator:
    """Evaluator that batches proposals for simultaneous evaluation.

    Provides:
    - Automatic batching of incoming proposals
    - Configurable batch size and timeout
    - Parallel batch evaluation
    - Throughput optimization
    """

    def __init__(
        self,
        evaluator_id: str = "batch",
        default_batch_size: int = 50,
        batch_timeout_ms: int = 1000,
        max_concurrent_batches: int = 4,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize batch evaluator.

        Args:
            evaluator_id: Unique evaluator identifier
            default_batch_size: Default maximum batch size
            batch_timeout_ms: Batch collection timeout
            max_concurrent_batches: Maximum concurrent batch evaluations
            merkle_chain: Merkle chain for audit trail
        """
        self.evaluator_id = evaluator_id
        self.default_batch_size = default_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_concurrent_batches = max_concurrent_batches
        self.merkle_chain = merkle_chain or MerkleChain()

        # Batch management
        self.batches: dict[str, EvaluationBatch] = {}
        self.current_batch: EvaluationBatch | None = None
        self._batch_counter = 0
        self._lock = threading.RLock()

        # Evaluation executor
        self._executor: Callable[[SandboxProposal], SandboxEvaluationResult] | None = None

        # Statistics
        self._total_proposals = 0
        self._total_batches = 0
        self._total_time_ms = 0.0

        # Batch timeout timer
        self._timer: threading.Timer | None = None

        # Log initialization
        self.merkle_chain.add_event(
            "batch_evaluator_initialized",
            {
                "evaluator_id": evaluator_id,
                "default_batch_size": default_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_executor(
        self,
        executor: Callable[[SandboxProposal], SandboxEvaluationResult],
    ) -> None:
        """Set the proposal executor function.

        Args:
            executor: Function to execute individual proposals
        """
        self._executor = executor

    def submit(
        self,
        proposal: SandboxProposal,
        immediate: bool = False,
    ) -> str:
        """Submit a proposal for batched evaluation.

        Args:
            proposal: Proposal to evaluate
            immediate: If True, evaluate immediately without batching

        Returns:
            Batch ID the proposal was added to
        """
        with self._lock:
            self._total_proposals += 1

            if immediate or proposal.priority == ProposalPriority.CRITICAL:
                # Create single-proposal batch and evaluate immediately
                batch = self._create_batch()
                batch.add_proposal(proposal)
                batch.status = BatchStatus.READY
                return batch.batch_id

            # Add to current batch
            if self.current_batch is None or self.current_batch.is_full:
                self.current_batch = self._create_batch()
                self._start_batch_timer()

            self.current_batch.add_proposal(proposal)

            # Check if batch is full
            if self.current_batch.is_full:
                self._cancel_batch_timer()
                self.current_batch.status = BatchStatus.READY

            return self.current_batch.batch_id

    def _create_batch(self) -> EvaluationBatch:
        """Create a new evaluation batch."""
        self._batch_counter += 1
        batch_id = f"batch_{self.evaluator_id}_{self._batch_counter:08d}"

        batch = EvaluationBatch(
            batch_id=batch_id,
            max_size=self.default_batch_size,
            timeout_ms=self.batch_timeout_ms,
        )

        self.batches[batch_id] = batch
        return batch

    def _start_batch_timer(self) -> None:
        """Start batch collection timeout timer."""
        self._cancel_batch_timer()
        self._timer = threading.Timer(
            self.batch_timeout_ms / 1000.0,
            self._on_batch_timeout,
        )
        self._timer.start()

    def _cancel_batch_timer(self) -> None:
        """Cancel batch collection timeout timer."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_batch_timeout(self) -> None:
        """Handle batch collection timeout."""
        with self._lock:
            if self.current_batch and self.current_batch.status == BatchStatus.COLLECTING:
                self.current_batch.status = BatchStatus.READY

    def get_ready_batches(self) -> list[EvaluationBatch]:
        """Get all batches ready for evaluation."""
        return [b for b in self.batches.values() if b.is_ready]

    def evaluate_batch(
        self,
        batch_id: str,
        executor: Callable[[SandboxProposal], SandboxEvaluationResult] | None = None,
    ) -> BatchResult:
        """Evaluate a batch of proposals.

        Args:
            batch_id: Batch to evaluate
            executor: Optional executor function (uses default if not provided)

        Returns:
            BatchResult with all evaluation results
        """
        with self._lock:
            if batch_id not in self.batches:
                raise ValueError(f"Batch {batch_id} not found")

            batch = self.batches[batch_id]
            if batch.status == BatchStatus.EVALUATING:
                raise RuntimeError(f"Batch {batch_id} already evaluating")

            batch.status = BatchStatus.EVALUATING
            batch.started_at = datetime.now(timezone.utc).isoformat()

        executor_func = executor or self._executor
        if executor_func is None:
            raise RuntimeError("No executor function configured")

        # Log batch evaluation start
        self.merkle_chain.add_event(
            "batch_evaluation_started",
            {
                "batch_id": batch_id,
                "proposal_count": len(batch.proposals),
            },
        )

        start_time = time.perf_counter()
        results: list[SandboxEvaluationResult] = []
        errors: list[str] = []

        try:
            for proposal in batch.proposals:
                try:
                    result = executor_func(proposal)
                    results.append(result)
                    batch.results.append(result)
                except Exception as e:
                    error_msg = f"Proposal {proposal.proposal_id} failed: {str(e)}"
                    errors.append(error_msg)

            batch.status = BatchStatus.COMPLETED
            success = len(errors) == 0

        except Exception as e:
            batch.status = BatchStatus.FAILED
            errors.append(f"Batch evaluation failed: {str(e)}")
            success = False

        batch.completed_at = datetime.now(timezone.utc).isoformat()
        total_time_ms = (time.perf_counter() - start_time) * 1000
        self._total_time_ms += total_time_ms
        self._total_batches += 1

        avg_time = total_time_ms / len(batch.proposals) if batch.proposals else 0
        throughput = len(results) / (total_time_ms / 1000) if total_time_ms > 0 else 0

        # Log batch evaluation completion
        self.merkle_chain.add_event(
            "batch_evaluation_completed",
            {
                "batch_id": batch_id,
                "success": success,
                "result_count": len(results),
                "total_time_ms": total_time_ms,
            },
        )

        return BatchResult(
            batch_id=batch_id,
            success=success,
            results=results,
            total_time_ms=total_time_ms,
            avg_time_per_proposal_ms=avg_time,
            throughput=throughput,
            merkle_proof=self.merkle_chain.get_chain_proof(),
            errors=errors,
        )

    def evaluate_all_ready(
        self,
        executor: Callable[[SandboxProposal], SandboxEvaluationResult] | None = None,
    ) -> list[BatchResult]:
        """Evaluate all ready batches.

        Args:
            executor: Optional executor function

        Returns:
            List of batch results
        """
        ready_batches = self.get_ready_batches()
        results: list[BatchResult] = []

        for batch in ready_batches:
            result = self.evaluate_batch(batch.batch_id, executor)
            results.append(result)

        return results

    def flush(self) -> EvaluationBatch | None:
        """Flush current batch, marking it ready for evaluation.

        Returns:
            Flushed batch if one was pending
        """
        with self._lock:
            self._cancel_batch_timer()

            if self.current_batch and self.current_batch.status == BatchStatus.COLLECTING:
                self.current_batch.status = BatchStatus.READY
                flushed = self.current_batch
                self.current_batch = None
                return flushed

            return None

    def get_batch(self, batch_id: str) -> EvaluationBatch | None:
        """Get batch by ID."""
        return self.batches.get(batch_id)

    def get_evaluator_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        status_counts: dict[str, int] = {}
        for batch in self.batches.values():
            status_counts[batch.status.value] = status_counts.get(batch.status.value, 0) + 1

        avg_batch_time = self._total_time_ms / self._total_batches if self._total_batches > 0 else 0
        overall_throughput = (
            self._total_proposals / (self._total_time_ms / 1000) if self._total_time_ms > 0 else 0
        )

        return {
            "evaluator_id": self.evaluator_id,
            "default_batch_size": self.default_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "total_batches": self._total_batches,
            "total_proposals": self._total_proposals,
            "total_time_ms": self._total_time_ms,
            "avg_batch_time_ms": avg_batch_time,
            "overall_throughput": overall_throughput,
            "batch_status_counts": status_counts,
            "current_batch_size": (len(self.current_batch.proposals) if self.current_batch else 0),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }

"""Proposal Queue for Asynchronous Artifact Queuing.

Implements async queuing of artifacts for the ReinjectionEvaluationEngine
without blocking production operations.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain

from qratum_asi.sandbox_platform.types import ProposalPriority, SandboxProposal


class QueuePriority(Enum):
    """Priority levels for queued items."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BATCH = "batch"


class QueuedProposalStatus(Enum):
    """Status of a queued proposal."""

    QUEUED = "queued"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedProposal:
    """Proposal queued for evaluation.

    Attributes:
        queue_id: Unique queue entry identifier
        proposal: The sandbox proposal
        priority: Queue priority
        status: Current status
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
        requires_approval: Whether dual-control approval is required
    """

    queue_id: str
    proposal: SandboxProposal
    priority: QueuePriority = QueuePriority.NORMAL
    status: QueuedProposalStatus = QueuedProposalStatus.QUEUED
    retry_count: int = 0
    max_retries: int = 3
    requires_approval: bool = False
    approval_status: str = "pending"
    enqueued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processed_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def compute_hash(self) -> str:
        """Compute hash of queued proposal."""
        content = {
            "queue_id": self.queue_id,
            "proposal_id": self.proposal.proposal_id,
            "priority": self.priority.value,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize queued proposal."""
        return {
            "queue_id": self.queue_id,
            "proposal_id": self.proposal.proposal_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status,
            "enqueued_at": self.enqueued_at,
            "processed_at": self.processed_at,
            "completed_at": self.completed_at,
            "has_result": self.result is not None,
            "error": self.error,
        }


class ProposalQueue:
    """Asynchronous proposal queue for sandbox evaluation.

    Provides:
    - Non-blocking proposal submission
    - Priority-based processing
    - Integration with ReinjectionEvaluationEngine
    - Dual-control approval integration
    """

    def __init__(
        self,
        queue_id: str = "proposal_queue",
        max_size: int = 10000,
        num_workers: int = 4,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize proposal queue.

        Args:
            queue_id: Unique queue identifier
            max_size: Maximum queue size
            num_workers: Number of processing workers
            merkle_chain: Merkle chain for audit trail
        """
        self.queue_id = queue_id
        self.max_size = max_size
        self.num_workers = num_workers
        self.merkle_chain = merkle_chain or MerkleChain()

        # Priority queues
        self._queues: dict[QueuePriority, queue.Queue[QueuedProposal]] = {
            QueuePriority.IMMEDIATE: queue.Queue(maxsize=max_size // 5),
            QueuePriority.HIGH: queue.Queue(maxsize=max_size // 4),
            QueuePriority.NORMAL: queue.Queue(maxsize=max_size // 3),
            QueuePriority.LOW: queue.Queue(maxsize=max_size // 5),
            QueuePriority.BATCH: queue.Queue(maxsize=max_size // 5),
        }

        # Tracking
        self._proposals: dict[str, QueuedProposal] = {}
        self._proposal_counter = 0
        self._lock = threading.RLock()

        # Workers
        self._workers: list[threading.Thread] = []
        self._is_running = False
        self._shutdown_event = threading.Event()

        # Processing callback
        self._processor: Callable[[QueuedProposal], dict[str, Any]] | None = None

        # Statistics
        self._enqueued_count = 0
        self._processed_count = 0
        self._failed_count = 0
        self._rejected_count = 0

        # Log initialization
        self.merkle_chain.add_event(
            "proposal_queue_initialized",
            {
                "queue_id": queue_id,
                "max_size": max_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_processor(
        self,
        processor: Callable[[QueuedProposal], dict[str, Any]],
    ) -> None:
        """Set the proposal processor function.

        Args:
            processor: Function to process proposals
        """
        self._processor = processor

    def enqueue(
        self,
        proposal: SandboxProposal,
        priority: QueuePriority = QueuePriority.NORMAL,
        requires_approval: bool = False,
    ) -> QueuedProposal | None:
        """Enqueue a proposal for processing.

        Args:
            proposal: Proposal to enqueue
            priority: Queue priority
            requires_approval: Whether approval is required

        Returns:
            QueuedProposal if enqueued, None if queue full
        """
        with self._lock:
            self._proposal_counter += 1
            queue_id = f"q_{self.queue_id}_{self._proposal_counter:010d}"

            queued = QueuedProposal(
                queue_id=queue_id,
                proposal=proposal,
                priority=priority,
                requires_approval=requires_approval,
            )

            target_queue = self._queues[priority]
            try:
                target_queue.put_nowait(queued)
                self._proposals[queue_id] = queued
                self._enqueued_count += 1

                # Log enqueue
                self.merkle_chain.add_event(
                    "proposal_enqueued",
                    {
                        "queue_id": queue_id,
                        "proposal_id": proposal.proposal_id,
                        "priority": priority.value,
                    },
                )

                return queued

            except queue.Full:
                return None

    def dequeue(self) -> QueuedProposal | None:
        """Dequeue highest priority proposal.

        Returns:
            QueuedProposal if available, None otherwise
        """
        # Process in priority order
        for priority in [
            QueuePriority.IMMEDIATE,
            QueuePriority.HIGH,
            QueuePriority.NORMAL,
            QueuePriority.LOW,
            QueuePriority.BATCH,
        ]:
            try:
                return self._queues[priority].get_nowait()
            except queue.Empty:
                continue
        return None

    def start(self) -> None:
        """Start queue processing workers."""
        if self._is_running:
            return

        self._is_running = True
        self._shutdown_event.clear()

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"proposal_queue_worker_{self.queue_id}_{i}",
            )
            self._workers.append(worker)
            worker.start()

        self.merkle_chain.add_event(
            "proposal_queue_started",
            {
                "queue_id": self.queue_id,
                "num_workers": self.num_workers,
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop queue processing workers.

        Args:
            timeout: Maximum time to wait
        """
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_event.set()

        for worker in self._workers:
            worker.join(timeout=timeout / len(self._workers))

        self._workers.clear()

        self.merkle_chain.add_event(
            "proposal_queue_stopped",
            {
                "queue_id": self.queue_id,
                "processed_count": self._processed_count,
            },
        )

    def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing proposals."""
        while not self._shutdown_event.is_set():
            proposal = self.dequeue()
            if proposal is None:
                time.sleep(0.01)
                continue

            try:
                self._process_proposal(proposal, worker_id)
            except Exception as e:
                proposal.status = QueuedProposalStatus.FAILED
                proposal.error = str(e)
                self._failed_count += 1

    def _process_proposal(self, queued: QueuedProposal, worker_id: int) -> None:
        """Process a queued proposal."""
        queued.status = QueuedProposalStatus.PROCESSING
        queued.processed_at = datetime.now(timezone.utc).isoformat()

        # Check if approval required
        if queued.requires_approval and queued.approval_status == "pending":
            queued.status = QueuedProposalStatus.AWAITING_APPROVAL
            # Re-queue for approval check
            self._queues[QueuePriority.LOW].put(queued)
            return

        if self._processor is None:
            queued.status = QueuedProposalStatus.FAILED
            queued.error = "No processor configured"
            return

        try:
            result = self._processor(queued)
            queued.result = result
            queued.status = QueuedProposalStatus.COMPLETED
            queued.completed_at = datetime.now(timezone.utc).isoformat()
            self._processed_count += 1

            # Log completion
            self.merkle_chain.add_event(
                "proposal_processed",
                {
                    "queue_id": queued.queue_id,
                    "proposal_id": queued.proposal.proposal_id,
                    "status": "completed",
                },
            )

        except Exception as e:
            queued.retry_count += 1
            if queued.retry_count < queued.max_retries:
                # Re-queue for retry
                queued.status = QueuedProposalStatus.QUEUED
                self._queues[QueuePriority.LOW].put(queued)
            else:
                queued.status = QueuedProposalStatus.FAILED
                queued.error = str(e)
                self._failed_count += 1

    def approve(self, queue_id: str, approved: bool, reason: str = "") -> bool:
        """Approve or reject a queued proposal.

        Args:
            queue_id: Queue ID to approve
            approved: Whether to approve
            reason: Approval/rejection reason

        Returns:
            True if approval processed
        """
        with self._lock:
            if queue_id not in self._proposals:
                return False

            queued = self._proposals[queue_id]
            if queued.status != QueuedProposalStatus.AWAITING_APPROVAL:
                return False

            if approved:
                queued.approval_status = "approved"
                queued.status = QueuedProposalStatus.APPROVED
                # Re-queue for processing
                self._queues[queued.priority].put(queued)
            else:
                queued.approval_status = "rejected"
                queued.status = QueuedProposalStatus.REJECTED
                queued.error = reason
                self._rejected_count += 1

            # Log approval
            self.merkle_chain.add_event(
                "proposal_approval",
                {
                    "queue_id": queue_id,
                    "approved": approved,
                    "reason": reason,
                },
            )

            return True

    def get_proposal(self, queue_id: str) -> QueuedProposal | None:
        """Get proposal by queue ID."""
        return self._proposals.get(queue_id)

    def get_pending_approvals(self) -> list[QueuedProposal]:
        """Get all proposals awaiting approval."""
        return [
            p for p in self._proposals.values()
            if p.status == QueuedProposalStatus.AWAITING_APPROVAL
        ]

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {p.value: q.qsize() for p, q in self._queues.items()}
        total_size = sum(queue_sizes.values())

        status_counts: dict[str, int] = {}
        for p in self._proposals.values():
            status_counts[p.status.value] = status_counts.get(p.status.value, 0) + 1

        return {
            "queue_id": self.queue_id,
            "is_running": self._is_running,
            "num_workers": self.num_workers,
            "total_size": total_size,
            "max_size": self.max_size,
            "queue_sizes": queue_sizes,
            "enqueued_count": self._enqueued_count,
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "rejected_count": self._rejected_count,
            "status_counts": status_counts,
            "pending_approvals": len(self.get_pending_approvals()),
        }

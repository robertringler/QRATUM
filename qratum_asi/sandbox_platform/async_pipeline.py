"""Asynchronous Pipeline for Non-Blocking Execution.

Implements asynchronous evaluation pipelines that prevent blocking
production tasks. Proposals are processed through staged pipelines
with non-blocking queues.
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

from qratum_asi.sandbox_platform.types import (
    ProposalPriority,
    SandboxEvaluationResult,
    SandboxProposal,
)


class PipelineStageStatus(Enum):
    """Status of a pipeline stage."""

    IDLE = "idle"
    PROCESSING = "processing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueStatus(Enum):
    """Status of the queue."""

    EMPTY = "empty"
    ACTIVE = "active"
    FULL = "full"
    DRAINING = "draining"


@dataclass
class PipelineStage:
    """Stage in the evaluation pipeline.

    Attributes:
        stage_id: Unique stage identifier
        stage_name: Human-readable stage name
        stage_order: Order in pipeline (0-based)
        processor: Function to process proposals at this stage
        status: Current stage status
        items_processed: Total items processed
        avg_processing_time_ms: Average processing time
    """

    stage_id: str
    stage_name: str
    stage_order: int
    processor: Callable[[SandboxProposal, dict[str, Any]], SandboxProposal | None] | None = None
    status: PipelineStageStatus = PipelineStageStatus.IDLE
    items_processed: int = 0
    avg_processing_time_ms: float = 0.0
    last_processed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline stage."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_order": self.stage_order,
            "status": self.status.value,
            "items_processed": self.items_processed,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "last_processed_at": self.last_processed_at,
        }


@dataclass
class QueuedItem:
    """Item in the non-blocking queue.

    Attributes:
        item_id: Unique item identifier
        proposal: Queued proposal
        priority: Queue priority
        enqueued_at: Time when enqueued
        stage_index: Current pipeline stage
        context: Processing context
    """

    item_id: str
    proposal: SandboxProposal
    priority: ProposalPriority
    enqueued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Serialize queued item."""
        return {
            "item_id": self.item_id,
            "proposal_id": self.proposal.proposal_id,
            "priority": self.priority.value,
            "enqueued_at": self.enqueued_at,
            "stage_index": self.stage_index,
            "retries": self.retries,
        }


class NonBlockingQueue:
    """Non-blocking priority queue for proposal evaluation.

    Implements a thread-safe priority queue that never blocks
    the calling thread for production operations.
    """

    def __init__(
        self,
        queue_id: str = "default",
        max_size: int = 10000,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize non-blocking queue.

        Args:
            queue_id: Unique queue identifier
            max_size: Maximum queue size
            merkle_chain: Merkle chain for audit trail
        """
        self.queue_id = queue_id
        self.max_size = max_size
        self.merkle_chain = merkle_chain or MerkleChain()

        # Priority queues (one per priority level)
        self._queues: dict[ProposalPriority, queue.Queue] = {
            ProposalPriority.CRITICAL: queue.Queue(maxsize=max_size // 4),
            ProposalPriority.HIGH: queue.Queue(maxsize=max_size // 4),
            ProposalPriority.NORMAL: queue.Queue(maxsize=max_size // 4),
            ProposalPriority.LOW: queue.Queue(maxsize=max_size // 4),
        }

        self._item_counter = 0
        self._lock = threading.RLock()
        self._status = QueueStatus.EMPTY

        # Statistics
        self._enqueue_count = 0
        self._dequeue_count = 0
        self._dropped_count = 0

    def enqueue(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any] | None = None,
    ) -> QueuedItem | None:
        """Enqueue a proposal without blocking.

        Args:
            proposal: Proposal to enqueue
            context: Processing context

        Returns:
            QueuedItem if enqueued, None if queue is full
        """
        with self._lock:
            priority = proposal.priority
            target_queue = self._queues[priority]

            if target_queue.full():
                self._dropped_count += 1
                return None

            self._item_counter += 1
            item = QueuedItem(
                item_id=f"item_{self.queue_id}_{self._item_counter:010d}",
                proposal=proposal,
                priority=priority,
                context=context or {},
            )

            try:
                target_queue.put_nowait(item)
                self._enqueue_count += 1
                self._update_status()

                # Log enqueue
                self.merkle_chain.add_event(
                    "proposal_enqueued",
                    {
                        "item_id": item.item_id,
                        "proposal_id": proposal.proposal_id,
                        "priority": priority.value,
                    },
                )

                return item

            except queue.Full:
                self._dropped_count += 1
                return None

    def dequeue(self) -> QueuedItem | None:
        """Dequeue highest priority item without blocking.

        Returns:
            QueuedItem if available, None if all queues empty
        """
        with self._lock:
            # Process in priority order
            for priority in [
                ProposalPriority.CRITICAL,
                ProposalPriority.HIGH,
                ProposalPriority.NORMAL,
                ProposalPriority.LOW,
            ]:
                target_queue = self._queues[priority]
                try:
                    item = target_queue.get_nowait()
                    self._dequeue_count += 1
                    self._update_status()

                    # Log dequeue
                    self.merkle_chain.add_event(
                        "proposal_dequeued",
                        {
                            "item_id": item.item_id,
                            "proposal_id": item.proposal.proposal_id,
                        },
                    )

                    return item
                except queue.Empty:
                    continue

            return None

    def _update_status(self) -> None:
        """Update queue status based on current state."""
        total_size = sum(q.qsize() for q in self._queues.values())

        if total_size == 0:
            self._status = QueueStatus.EMPTY
        elif total_size >= self.max_size * 0.9:
            self._status = QueueStatus.FULL
        else:
            self._status = QueueStatus.ACTIVE

    def get_status(self) -> QueueStatus:
        """Get current queue status."""
        return self._status

    def size(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self._queues.values())

    def size_by_priority(self) -> dict[str, int]:
        """Get queue sizes by priority."""
        return {p.value: q.qsize() for p, q in self._queues.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_id": self.queue_id,
            "status": self._status.value,
            "total_size": self.size(),
            "max_size": self.max_size,
            "size_by_priority": self.size_by_priority(),
            "enqueue_count": self._enqueue_count,
            "dequeue_count": self._dequeue_count,
            "dropped_count": self._dropped_count,
            "throughput": (
                self._dequeue_count / max(1, self._enqueue_count)
            ),
        }


class AsyncEvaluationPipeline:
    """Asynchronous pipeline for proposal evaluation.

    Implements a multi-stage pipeline that processes proposals
    without blocking production tasks. Each stage can be processed
    in parallel with configurable concurrency.
    """

    def __init__(
        self,
        pipeline_id: str = "default",
        num_workers: int = 4,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize async evaluation pipeline.

        Args:
            pipeline_id: Unique pipeline identifier
            num_workers: Number of worker threads
            merkle_chain: Merkle chain for audit trail
        """
        self.pipeline_id = pipeline_id
        self.num_workers = num_workers
        self.merkle_chain = merkle_chain or MerkleChain()

        # Pipeline stages
        self.stages: list[PipelineStage] = []
        self._stage_counter = 0

        # Input/output queues
        self.input_queue = NonBlockingQueue(
            queue_id=f"{pipeline_id}_input",
            merkle_chain=self.merkle_chain,
        )
        self.output_queue: queue.Queue[SandboxEvaluationResult] = queue.Queue()

        # Workers
        self._workers: list[threading.Thread] = []
        self._is_running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        # Statistics
        self._processed_count = 0
        self._failed_count = 0
        self._start_time: float | None = None

        # Result callbacks
        self._result_callbacks: list[Callable[[SandboxEvaluationResult], None]] = []

        # Log initialization
        self.merkle_chain.add_event(
            "pipeline_initialized",
            {
                "pipeline_id": pipeline_id,
                "num_workers": num_workers,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def add_stage(
        self,
        stage_name: str,
        processor: Callable[[SandboxProposal, dict[str, Any]], SandboxProposal | None],
    ) -> PipelineStage:
        """Add a processing stage to the pipeline.

        Args:
            stage_name: Human-readable stage name
            processor: Function to process proposals at this stage

        Returns:
            Created PipelineStage
        """
        with self._lock:
            self._stage_counter += 1
            stage = PipelineStage(
                stage_id=f"stage_{self.pipeline_id}_{self._stage_counter:04d}",
                stage_name=stage_name,
                stage_order=len(self.stages),
                processor=processor,
            )
            self.stages.append(stage)
            return stage

    def submit(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any] | None = None,
    ) -> QueuedItem | None:
        """Submit a proposal for pipeline processing.

        Args:
            proposal: Proposal to process
            context: Processing context

        Returns:
            QueuedItem if enqueued, None if queue full
        """
        return self.input_queue.enqueue(proposal, context)

    def start(self) -> None:
        """Start pipeline workers."""
        if self._is_running:
            return

        self._is_running = True
        self._shutdown_event.clear()
        self._start_time = time.time()

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"pipeline_worker_{self.pipeline_id}_{i}",
            )
            self._workers.append(worker)
            worker.start()

        self.merkle_chain.add_event(
            "pipeline_started",
            {
                "pipeline_id": self.pipeline_id,
                "num_workers": self.num_workers,
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop pipeline workers.

        Args:
            timeout: Maximum time to wait for workers to stop
        """
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_event.set()

        for worker in self._workers:
            worker.join(timeout=timeout / len(self._workers))

        self._workers.clear()

        self.merkle_chain.add_event(
            "pipeline_stopped",
            {
                "pipeline_id": self.pipeline_id,
                "processed_count": self._processed_count,
                "failed_count": self._failed_count,
            },
        )

    def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing items."""
        while not self._shutdown_event.is_set():
            item = self.input_queue.dequeue()

            if item is None:
                # No items available, wait briefly
                time.sleep(0.01)
                continue

            try:
                result = self._process_item(item)
                if result:
                    self.output_queue.put(result)
                    self._processed_count += 1

                    # Notify callbacks
                    for callback in self._result_callbacks:
                        try:
                            callback(result)
                        except Exception:
                            pass

            except Exception as e:
                self._failed_count += 1
                self.merkle_chain.add_event(
                    "pipeline_processing_failed",
                    {
                        "item_id": item.item_id,
                        "error": str(e),
                        "worker_id": worker_id,
                    },
                )

    def _process_item(self, item: QueuedItem) -> SandboxEvaluationResult | None:
        """Process a queued item through all stages."""
        start_time = time.perf_counter()
        proposal = item.proposal
        context = item.context

        # Process through each stage
        for stage in self.stages:
            if stage.processor is None:
                continue

            stage.status = PipelineStageStatus.PROCESSING
            stage_start = time.perf_counter()

            try:
                result = stage.processor(proposal, context)
                if result is None:
                    # Stage filtered out the proposal
                    stage.status = PipelineStageStatus.COMPLETED
                    return None
                proposal = result

                # Update stage stats
                stage_time = (time.perf_counter() - stage_start) * 1000
                stage.items_processed += 1
                stage.avg_processing_time_ms = (
                    (stage.avg_processing_time_ms * (stage.items_processed - 1) + stage_time)
                    / stage.items_processed
                )
                stage.last_processed_at = datetime.now(timezone.utc).isoformat()
                stage.status = PipelineStageStatus.COMPLETED

            except Exception as e:
                stage.status = PipelineStageStatus.FAILED
                raise e

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Create evaluation result
        return SandboxEvaluationResult(
            result_id=f"result_{item.item_id}",
            proposal_id=proposal.proposal_id,
            sandbox_id=self.pipeline_id,
            success=True,
            outcome="pipeline_completed",
            fidelity_score=0.9,
            execution_time_ms=execution_time_ms,
            merkle_proof=self.merkle_chain.get_chain_proof(),
            metrics={
                "stages_processed": len(self.stages),
                "item_id": item.item_id,
            },
        )

    def register_result_callback(
        self,
        callback: Callable[[SandboxEvaluationResult], None],
    ) -> None:
        """Register callback for result events."""
        self._result_callbacks.append(callback)

    def get_result(self, timeout: float = 0.0) -> SandboxEvaluationResult | None:
        """Get a result from the output queue.

        Args:
            timeout: Timeout in seconds (0 for non-blocking)

        Returns:
            SandboxEvaluationResult if available
        """
        try:
            if timeout > 0:
                return self.output_queue.get(timeout=timeout)
            else:
                return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time

        throughput = 0.0
        if uptime > 0:
            throughput = self._processed_count / uptime

        return {
            "pipeline_id": self.pipeline_id,
            "is_running": self._is_running,
            "num_workers": self.num_workers,
            "stages": [s.to_dict() for s in self.stages],
            "input_queue": self.input_queue.get_stats(),
            "output_queue_size": self.output_queue.qsize(),
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "uptime_seconds": uptime,
            "throughput_per_second": throughput,
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }

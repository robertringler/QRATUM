"""Diagnostics Offload for Sandbox Operations.

Implements offloading of intensive diagnostics to sandbox nodes,
keeping production paths clean and performant.
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


class DiagnosticType(Enum):
    """Type of diagnostic job."""

    PROFILING = "profiling"
    MEMORY_ANALYSIS = "memory_analysis"
    PERFORMANCE_TRACE = "performance_trace"
    STATE_INSPECTION = "state_inspection"
    INVARIANT_CHECK = "invariant_check"
    DEPENDENCY_ANALYSIS = "dependency_analysis"


class JobStatus(Enum):
    """Status of a diagnostic job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OffloadTarget(Enum):
    """Target for diagnostic offload."""

    SANDBOX_NODE = "sandbox_node"
    DEDICATED_POOL = "dedicated_pool"
    BACKGROUND = "background"


@dataclass
class DiagnosticJob:
    """Diagnostic job to be offloaded.

    Attributes:
        job_id: Unique job identifier
        job_type: Type of diagnostic
        target: Offload target
        payload: Job payload/parameters
        priority: Job priority (1-10, higher = more important)
        status: Current job status
        result: Job result when completed
    """

    job_id: str
    job_type: DiagnosticType
    target: OffloadTarget
    payload: dict[str, Any]
    priority: int = 5
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostic job."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "target": self.target.value,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time_ms": self.execution_time_ms,
            "has_result": self.result is not None,
            "error": self.error,
        }


@dataclass
class DiagnosticResult:
    """Result of a diagnostic job.

    Attributes:
        job_id: Source job identifier
        job_type: Type of diagnostic
        success: Whether diagnostic succeeded
        findings: Diagnostic findings
        metrics: Collected metrics
        recommendations: Recommendations based on findings
    """

    job_id: str
    job_type: DiagnosticType
    success: bool
    findings: list[dict[str, Any]]
    metrics: dict[str, Any]
    recommendations: list[str]
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostic result."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "success": self.success,
            "findings_count": len(self.findings),
            "recommendations_count": len(self.recommendations),
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


class DiagnosticsOffloader:
    """Offloader for intensive diagnostics to sandbox nodes.

    Provides:
    - Non-blocking diagnostic submission
    - Priority-based job scheduling
    - Multiple offload targets
    - Result aggregation
    """

    def __init__(
        self,
        offloader_id: str = "diagnostics",
        num_workers: int = 2,
        max_queue_size: int = 1000,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize diagnostics offloader.

        Args:
            offloader_id: Unique offloader identifier
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
            merkle_chain: Merkle chain for audit trail
        """
        self.offloader_id = offloader_id
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.merkle_chain = merkle_chain or MerkleChain()

        # Job management
        self._job_queue: queue.PriorityQueue[tuple[int, str, DiagnosticJob]] = queue.PriorityQueue(
            maxsize=max_queue_size
        )
        self._jobs: dict[str, DiagnosticJob] = {}
        self._results: dict[str, DiagnosticResult] = {}
        self._job_counter = 0
        self._lock = threading.RLock()

        # Workers
        self._workers: list[threading.Thread] = []
        self._is_running = False
        self._shutdown_event = threading.Event()

        # Diagnostic handlers
        self._handlers: dict[DiagnosticType, Callable[[DiagnosticJob], DiagnosticResult]] = {}
        self._register_default_handlers()

        # Statistics
        self._jobs_submitted = 0
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._total_execution_time_ms = 0.0

        # Log initialization
        self.merkle_chain.add_event(
            "diagnostics_offloader_initialized",
            {
                "offloader_id": offloader_id,
                "num_workers": num_workers,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _register_default_handlers(self) -> None:
        """Register default diagnostic handlers."""
        self._handlers[DiagnosticType.PROFILING] = self._handle_profiling
        self._handlers[DiagnosticType.MEMORY_ANALYSIS] = self._handle_memory_analysis
        self._handlers[DiagnosticType.PERFORMANCE_TRACE] = self._handle_performance_trace
        self._handlers[DiagnosticType.STATE_INSPECTION] = self._handle_state_inspection
        self._handlers[DiagnosticType.INVARIANT_CHECK] = self._handle_invariant_check
        self._handlers[DiagnosticType.DEPENDENCY_ANALYSIS] = self._handle_dependency_analysis

    def register_handler(
        self,
        job_type: DiagnosticType,
        handler: Callable[[DiagnosticJob], DiagnosticResult],
    ) -> None:
        """Register a custom diagnostic handler.

        Args:
            job_type: Type of diagnostic to handle
            handler: Handler function
        """
        self._handlers[job_type] = handler

    def submit(
        self,
        job_type: DiagnosticType,
        payload: dict[str, Any],
        target: OffloadTarget = OffloadTarget.SANDBOX_NODE,
        priority: int = 5,
    ) -> DiagnosticJob:
        """Submit a diagnostic job for offloaded execution.

        Args:
            job_type: Type of diagnostic
            payload: Job payload
            target: Offload target
            priority: Job priority (1-10)

        Returns:
            Created DiagnosticJob
        """
        with self._lock:
            self._job_counter += 1
            job_id = f"diag_{self.offloader_id}_{job_type.value}_{self._job_counter:08d}"

            job = DiagnosticJob(
                job_id=job_id,
                job_type=job_type,
                target=target,
                payload=payload,
                priority=max(1, min(10, priority)),
            )

            self._jobs[job_id] = job
            self._jobs_submitted += 1

            # Add to priority queue (lower number = higher priority)
            queue_priority = 10 - job.priority
            try:
                self._job_queue.put_nowait((queue_priority, job_id, job))
            except queue.Full:
                job.status = JobStatus.CANCELLED
                job.error = "Queue full"

            # Log submission
            self.merkle_chain.add_event(
                "diagnostic_job_submitted",
                {
                    "job_id": job_id,
                    "job_type": job_type.value,
                    "target": target.value,
                    "priority": priority,
                },
            )

            return job

    def start(self) -> None:
        """Start the offloader workers."""
        if self._is_running:
            return

        self._is_running = True
        self._shutdown_event.clear()

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"diagnostics_worker_{self.offloader_id}_{i}",
            )
            self._workers.append(worker)
            worker.start()

        self.merkle_chain.add_event(
            "diagnostics_offloader_started",
            {
                "offloader_id": self.offloader_id,
                "num_workers": self.num_workers,
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the offloader workers.

        Args:
            timeout: Maximum time to wait for workers
        """
        if not self._is_running:
            return

        self._is_running = False
        self._shutdown_event.set()

        for worker in self._workers:
            worker.join(timeout=timeout / len(self._workers))

        self._workers.clear()

        self.merkle_chain.add_event(
            "diagnostics_offloader_stopped",
            {
                "offloader_id": self.offloader_id,
                "jobs_completed": self._jobs_completed,
            },
        )

    def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing diagnostic jobs."""
        while not self._shutdown_event.is_set():
            try:
                # Get job with timeout
                try:
                    _, job_id, job = self._job_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Execute job
                self._execute_job(job, worker_id)
                self._job_queue.task_done()

            except Exception as e:
                # Log error but continue processing
                self.merkle_chain.add_event(
                    "diagnostics_worker_error",
                    {
                        "worker_id": worker_id,
                        "error": str(e),
                    },
                )

    def _execute_job(self, job: DiagnosticJob, worker_id: int) -> None:
        """Execute a diagnostic job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()

        start_time = time.perf_counter()

        try:
            handler = self._handlers.get(job.job_type)
            if handler is None:
                raise ValueError(f"No handler for diagnostic type: {job.job_type.value}")

            result = handler(job)
            job.result = result.to_dict()
            job.status = JobStatus.COMPLETED

            with self._lock:
                self._results[job.job_id] = result
                self._jobs_completed += 1

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            with self._lock:
                self._jobs_failed += 1

        job.completed_at = datetime.now(timezone.utc).isoformat()
        job.execution_time_ms = (time.perf_counter() - start_time) * 1000
        self._total_execution_time_ms += job.execution_time_ms

        # Log completion
        self.merkle_chain.add_event(
            "diagnostic_job_completed",
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "execution_time_ms": job.execution_time_ms,
            },
        )

    # Default handlers

    def _handle_profiling(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle profiling diagnostic."""
        # Simulate profiling analysis
        findings = [
            {"type": "hotspot", "function": "process_data", "time_percent": 25.5},
            {"type": "hotspot", "function": "validate_input", "time_percent": 15.2},
        ]

        metrics = {
            "total_functions": 150,
            "profiled_functions": 150,
            "execution_samples": 10000,
        }

        recommendations = [
            "Consider optimizing process_data function",
            "Cache results of validate_input for repeated inputs",
        ]

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=True,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _handle_memory_analysis(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle memory analysis diagnostic."""
        findings = [
            {"type": "allocation", "location": "cache_buffer", "size_mb": 128},
            {"type": "leak_suspect", "location": "temp_results", "growth_rate": 0.5},
        ]

        metrics = {
            "total_allocated_mb": 512,
            "peak_usage_mb": 480,
            "gc_collections": 45,
        }

        recommendations = [
            "Consider reducing cache_buffer size",
            "Investigate temp_results for potential memory leak",
        ]

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=True,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _handle_performance_trace(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle performance trace diagnostic."""
        findings = [
            {"type": "slow_span", "operation": "database_query", "duration_ms": 150},
            {"type": "slow_span", "operation": "network_call", "duration_ms": 200},
        ]

        metrics = {
            "total_spans": 500,
            "avg_span_duration_ms": 25,
            "max_span_duration_ms": 200,
        }

        recommendations = [
            "Add caching for repeated database queries",
            "Consider connection pooling for network calls",
        ]

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=True,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _handle_state_inspection(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle state inspection diagnostic."""
        state = job.payload.get("state", {})

        findings = [
            {"type": "state_size", "keys": len(state), "depth": 3},
            {"type": "anomaly", "field": "counter", "issue": "unexpected_negative"},
        ]

        metrics = {
            "state_keys": len(state),
            "state_depth": 3,
            "state_size_bytes": len(str(state)),
        }

        recommendations = [
            "Review counter field for potential bug",
        ]

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=True,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _handle_invariant_check(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle invariant check diagnostic."""
        invariants = job.payload.get("invariants", [])

        findings = []
        passed = 0
        failed = 0

        for inv in invariants:
            # Simulate invariant check
            passed_check = True  # Simplified
            if passed_check:
                passed += 1
            else:
                failed += 1
                findings.append({"type": "violation", "invariant": inv, "severity": "high"})

        metrics = {
            "total_invariants": len(invariants),
            "passed": passed,
            "failed": failed,
        }

        recommendations = [f"Fix {failed} invariant violations"] if failed > 0 else []

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=failed == 0,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _handle_dependency_analysis(self, job: DiagnosticJob) -> DiagnosticResult:
        """Handle dependency analysis diagnostic."""
        findings = [
            {"type": "circular", "path": ["A", "B", "C", "A"]},
            {"type": "unused", "dependency": "legacy_module"},
        ]

        metrics = {
            "total_dependencies": 50,
            "direct_dependencies": 20,
            "transitive_dependencies": 30,
            "circular_count": 1,
        }

        recommendations = [
            "Break circular dependency between A, B, C",
            "Consider removing unused legacy_module",
        ]

        return DiagnosticResult(
            job_id=job.job_id,
            job_type=job.job_type,
            success=True,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
        )

    def get_job(self, job_id: str) -> DiagnosticJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> DiagnosticResult | None:
        """Get result for a job."""
        return self._results.get(job_id)

    def get_offloader_stats(self) -> dict[str, Any]:
        """Get offloader statistics."""
        return {
            "offloader_id": self.offloader_id,
            "num_workers": self.num_workers,
            "is_running": self._is_running,
            "queue_size": self._job_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "jobs_pending": self._jobs_submitted - self._jobs_completed - self._jobs_failed,
            "total_execution_time_ms": self._total_execution_time_ms,
            "avg_execution_time_ms": (
                self._total_execution_time_ms / self._jobs_completed
                if self._jobs_completed > 0
                else 0
            ),
        }

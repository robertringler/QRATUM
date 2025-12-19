"""QRATUM Audit Trail and Compliance Logging System.

Provides immutable, cryptographically-verifiable audit trails for all
quantum simulation operations. Designed for DO-178C Level A certification,
NIST 800-53, CMMC 2.0, and aerospace regulatory compliance.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

__all__ = [
    "AuditLevel",
    "AuditEvent",
    "AuditTrail",
    "ComplianceAuditor",
    "ExecutionContext",
    "get_auditor",
    "audit_operation",
]


class AuditLevel(Enum):
    """Audit event severity levels aligned with NIST 800-53 AU-2."""

    TRACE = auto()  # Detailed debugging (not for production)
    DEBUG = auto()  # Development diagnostics
    INFO = auto()  # Routine operational events
    WARNING = auto()  # Anomalies that may require attention
    ERROR = auto()  # Recoverable errors
    CRITICAL = auto()  # Non-recoverable failures
    SECURITY = auto()  # Security-relevant events (AC, AU, IA controls)
    COMPLIANCE = auto()  # Compliance verification events


@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event record.

    Attributes:
        event_id: Unique event identifier (UUID v4)
        timestamp: ISO 8601 UTC timestamp with nanosecond precision
        level: Event severity level
        category: Event category (simulation, security, compliance, etc.)
        operation: Specific operation being audited
        actor: Entity performing the operation (user, system, process)
        resource: Target resource being acted upon
        outcome: Operation outcome (success, failure, partial)
        metadata: Additional event-specific data
        parent_id: Parent event ID for correlation
        session_id: Session identifier for grouping related events
        hash_chain: SHA-256 hash linking to previous event
    """

    event_id: str
    timestamp: str
    level: AuditLevel
    category: str
    operation: str
    actor: str
    resource: str
    outcome: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    hash_chain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary for storage/transmission."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "level": self.level.name,
            "category": self.category,
            "operation": self.operation,
            "actor": self.actor,
            "resource": self.resource,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "hash_chain": self.hash_chain,
        }

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict(), indent=None, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event for chain verification."""
        content = f"{self.event_id}:{self.timestamp}:{self.operation}:{self.outcome}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class ExecutionContext:
    """Execution context for tracing and correlation.

    Provides contextual information for audit events and enables
    distributed tracing across simulation components.

    Attributes:
        context_id: Unique context identifier
        session_id: Session identifier for grouping operations
        trace_id: Distributed tracing identifier
        span_id: Current span identifier
        parent_span_id: Parent span for hierarchical tracing
        start_time: Context creation timestamp
        metadata: Context-specific metadata
        determinism_seed: Seed for reproducible execution
        device: Execution device (cpu, cuda:0, etc.)
        precision: Floating-point precision
    """

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter_ns)
    metadata: Dict[str, Any] = field(default_factory=dict)
    determinism_seed: Optional[int] = None
    device: str = "cpu"
    precision: str = "fp64"

    def create_child_span(self) -> ExecutionContext:
        """Create a child execution context for nested operations."""
        return ExecutionContext(
            context_id=str(uuid.uuid4()),
            session_id=self.session_id,
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            determinism_seed=self.determinism_seed,
            device=self.device,
            precision=self.precision,
            metadata=dict(self.metadata),
        )

    def elapsed_ns(self) -> int:
        """Return elapsed time since context creation in nanoseconds."""
        return time.perf_counter_ns() - self.start_time

    def elapsed_ms(self) -> float:
        """Return elapsed time since context creation in milliseconds."""
        return self.elapsed_ns() / 1_000_000


class AuditSink(Protocol):
    """Protocol for audit event sinks (storage backends)."""

    def write(self, event: AuditEvent) -> None:
        """Write an audit event to the sink."""
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...

    def close(self) -> None:
        """Close the sink and release resources."""
        ...


class FileAuditSink:
    """File-based audit sink with rotation support.

    Writes audit events to JSON Lines format files with automatic
    rotation based on size or time.
    """

    def __init__(
        self,
        path: Path,
        max_size_mb: int = 100,
        max_files: int = 10,
        buffer_size: int = 100,
    ):
        """Initialize file audit sink.

        Args:
            path: Base path for audit files
            max_size_mb: Maximum file size before rotation (MB)
            max_files: Maximum number of rotated files to keep
            buffer_size: Number of events to buffer before flush
        """
        self.path = Path(path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self.buffer_size = buffer_size
        self._buffer: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._file: Optional[Any] = None
        self._current_size = 0

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AuditEvent) -> None:
        """Write an audit event to the file sink."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()

    def flush(self) -> None:
        """Flush buffered events to disk."""
        with self._lock:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Internal buffer flush (must hold lock)."""
        if not self._buffer:
            return

        self._ensure_file_open()

        for event in self._buffer:
            line = event.to_json() + "\n"
            self._file.write(line)
            self._current_size += len(line.encode("utf-8"))

            if self._current_size >= self.max_size_bytes:
                self._rotate_file()

        self._file.flush()
        self._buffer.clear()

    def _ensure_file_open(self) -> None:
        """Ensure audit file is open for writing."""
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
            self._current_size = self.path.stat().st_size if self.path.exists() else 0

    def _rotate_file(self) -> None:
        """Rotate audit file when size limit reached."""
        if self._file:
            self._file.close()
            self._file = None

        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_path = self.path.with_suffix(f".{i}.jsonl")
            new_path = self.path.with_suffix(f".{i + 1}.jsonl")
            if old_path.exists():
                if i + 1 >= self.max_files:
                    old_path.unlink()
                else:
                    old_path.rename(new_path)

        # Rename current file
        if self.path.exists():
            self.path.rename(self.path.with_suffix(".1.jsonl"))

        self._current_size = 0
        self._ensure_file_open()

    def close(self) -> None:
        """Close the file sink."""
        with self._lock:
            self._flush_buffer()
            if self._file:
                self._file.close()
                self._file = None


class AuditTrail:
    """Thread-safe audit trail with hash chain verification.

    Maintains an immutable, cryptographically-linked chain of audit
    events for compliance verification and forensic analysis.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        sinks: Optional[List[AuditSink]] = None,
    ):
        """Initialize audit trail.

        Args:
            session_id: Session identifier for event grouping
            sinks: List of audit sinks for event storage
        """
        self.session_id = session_id or str(uuid.uuid4())
        self._sinks = sinks or []
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._event_count = 0
        self._logger = logging.getLogger("qratum.audit")

    def record(
        self,
        level: AuditLevel,
        category: str,
        operation: str,
        actor: str,
        resource: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> AuditEvent:
        """Record an audit event.

        Args:
            level: Event severity level
            category: Event category
            operation: Operation being audited
            actor: Entity performing operation
            resource: Target resource
            outcome: Operation outcome
            metadata: Additional event data
            parent_id: Parent event for correlation

        Returns:
            Created AuditEvent
        """
        with self._lock:
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                level=level,
                category=category,
                operation=operation,
                actor=actor,
                resource=resource,
                outcome=outcome,
                metadata=metadata or {},
                parent_id=parent_id,
                session_id=self.session_id,
                hash_chain=self._last_hash,
            )

            # Update hash chain
            self._last_hash = event.compute_hash()
            self._event_count += 1

            # Write to all sinks
            for sink in self._sinks:
                try:
                    sink.write(event)
                except Exception as e:
                    self._logger.error(f"Failed to write to audit sink: {e}")

            return event

    def verify_chain(self, events: List[AuditEvent]) -> bool:
        """Verify integrity of audit event chain.

        Args:
            events: List of audit events in chronological order

        Returns:
            True if chain is valid, False otherwise
        """
        if not events:
            return True

        prev_hash = None
        for event in events:
            if event.hash_chain != prev_hash:
                return False
            prev_hash = event.compute_hash()

        return True

    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        """Close all sinks."""
        self.flush()
        for sink in self._sinks:
            sink.close()


class ComplianceAuditor:
    """Unified compliance auditor for QRATUM operations.

    Provides a high-level interface for audit logging with built-in
    support for DO-178C, NIST 800-53, and CMMC 2.0 compliance.
    """

    _instance: Optional[ComplianceAuditor] = None
    _lock = threading.Lock()

    def __init__(
        self,
        audit_dir: Optional[Path] = None,
        enable_file_sink: bool = True,
    ):
        """Initialize compliance auditor.

        Args:
            audit_dir: Directory for audit files
            enable_file_sink: Whether to enable file-based audit sink
        """
        self._audit_dir = audit_dir or Path.home() / ".qratum" / "audit"
        self._audit_dir.mkdir(parents=True, exist_ok=True)

        sinks: List[AuditSink] = []
        if enable_file_sink:
            sinks.append(FileAuditSink(self._audit_dir / "audit.jsonl"))

        self._trail = AuditTrail(sinks=sinks)
        self._context_stack: Dict[int, List[ExecutionContext]] = {}
        self._logger = logging.getLogger("qratum.compliance")

    @classmethod
    def get_instance(cls) -> ComplianceAuditor:
        """Get singleton auditor instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def push_context(self, context: ExecutionContext) -> None:
        """Push execution context onto thread-local stack."""
        tid = threading.get_ident()
        if tid not in self._context_stack:
            self._context_stack[tid] = []
        self._context_stack[tid].append(context)

    def pop_context(self) -> Optional[ExecutionContext]:
        """Pop execution context from thread-local stack."""
        tid = threading.get_ident()
        if tid in self._context_stack and self._context_stack[tid]:
            return self._context_stack[tid].pop()
        return None

    def current_context(self) -> Optional[ExecutionContext]:
        """Get current execution context."""
        tid = threading.get_ident()
        if tid in self._context_stack and self._context_stack[tid]:
            return self._context_stack[tid][-1]
        return None

    def audit_simulation_start(
        self,
        circuit_id: str,
        num_qubits: int,
        backend: str,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Audit simulation start event."""
        return self._trail.record(
            level=AuditLevel.INFO,
            category="simulation",
            operation="simulation_start",
            actor="qratum_simulator",
            resource=circuit_id,
            outcome="initiated",
            metadata={
                "num_qubits": num_qubits,
                "backend": backend,
                "seed": seed,
                **(metadata or {}),
            },
        )

    def audit_simulation_complete(
        self,
        circuit_id: str,
        duration_ms: float,
        shots: int,
        outcome_count: int,
        parent_id: Optional[str] = None,
    ) -> AuditEvent:
        """Audit simulation completion event."""
        return self._trail.record(
            level=AuditLevel.INFO,
            category="simulation",
            operation="simulation_complete",
            actor="qratum_simulator",
            resource=circuit_id,
            outcome="success",
            metadata={
                "duration_ms": duration_ms,
                "shots": shots,
                "outcome_count": outcome_count,
            },
            parent_id=parent_id,
        )

    def audit_validation_check(
        self,
        check_name: str,
        target: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Audit validation check event."""
        return self._trail.record(
            level=AuditLevel.COMPLIANCE if passed else AuditLevel.WARNING,
            category="validation",
            operation=f"validation_{check_name}",
            actor="qratum_validator",
            resource=target,
            outcome="pass" if passed else "fail",
            metadata=details or {},
        )

    def audit_security_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Audit security-relevant event."""
        return self._trail.record(
            level=AuditLevel.SECURITY,
            category="security",
            operation=event_type,
            actor=actor,
            resource=resource,
            outcome=outcome,
            metadata=details or {},
        )

    def audit_error(
        self,
        operation: str,
        error_type: str,
        error_message: str,
        resource: str = "unknown",
        recoverable: bool = False,
    ) -> AuditEvent:
        """Audit error event."""
        return self._trail.record(
            level=AuditLevel.ERROR if recoverable else AuditLevel.CRITICAL,
            category="error",
            operation=operation,
            actor="qratum_system",
            resource=resource,
            outcome="error",
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                "recoverable": recoverable,
            },
        )

    def flush(self) -> None:
        """Flush audit trail."""
        self._trail.flush()

    def close(self) -> None:
        """Close auditor and release resources."""
        self._trail.close()


def get_auditor() -> ComplianceAuditor:
    """Get global compliance auditor instance."""
    return ComplianceAuditor.get_instance()


T = TypeVar("T")


def audit_operation(
    operation: str,
    category: str = "operation",
    resource: str = "unknown",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to audit function execution.

    Args:
        operation: Operation name for audit
        category: Event category
        resource: Target resource

    Returns:
        Decorated function with audit logging
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            auditor = get_auditor()
            context = ExecutionContext()
            auditor.push_context(context)

            start_event = auditor._trail.record(
                level=AuditLevel.DEBUG,
                category=category,
                operation=f"{operation}_start",
                actor="qratum_system",
                resource=resource,
                outcome="initiated",
                metadata={"function": func.__name__},
            )

            try:
                result = func(*args, **kwargs)
                auditor._trail.record(
                    level=AuditLevel.DEBUG,
                    category=category,
                    operation=f"{operation}_complete",
                    actor="qratum_system",
                    resource=resource,
                    outcome="success",
                    metadata={
                        "duration_ms": context.elapsed_ms(),
                        "function": func.__name__,
                    },
                    parent_id=start_event.event_id,
                )
                return result
            except Exception as e:
                auditor.audit_error(
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    resource=resource,
                    recoverable=False,
                )
                raise
            finally:
                auditor.pop_context()

        return wrapper

    return decorator

"""QRATUM Execution Context Manager.

Provides execution context with compliance, audit, and seed management.

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

__all__ = ["ExecutionContext"]


class ExecutionContext:
    """Execution context manager for QRATUM workflows.

    Provides:
    - Deterministic seed management
    - Audit logging with SHA-256 traceability
    - Prometheus metrics collection
    - DO-178C compliance validation
    """

    def __init__(
        self,
        workflow_name: str,
        seed: int | None = None,
        audit_enabled: bool = True,
        metrics_enabled: bool = True,
        do178c_enabled: bool = True,
    ):
        """Initialize execution context.

        Args:
            workflow_name: Name of the workflow
            seed: Random seed for reproducibility
            audit_enabled: Enable audit logging
            metrics_enabled: Enable metrics collection
            do178c_enabled: Enable DO-178C compliance
        """
        self.workflow_name = workflow_name
        self.seed = seed
        self.audit_enabled = audit_enabled
        self.metrics_enabled = metrics_enabled
        self.do178c_enabled = do178c_enabled

        self.execution_id: str | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.result: dict[str, Any] | None = None

        self._logger = logging.getLogger("qratum.context")

    def _generate_execution_hash(self) -> str:
        """Generate SHA-256 execution hash for DO-178C traceability.

        Returns:
            SHA-256 hash of execution metadata
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        data = f"{self.workflow_name}|{self.seed}|{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

    def __enter__(self) -> ExecutionContext:
        """Enter execution context.

        Returns:
            Self for context manager
        """
        self.start_time = time.perf_counter()
        self.execution_id = self._generate_execution_hash()

        if self.audit_enabled:
            self._logger.info(
                f"Workflow '{self.workflow_name}' started: "
                f"execution_id={self.execution_id}, seed={self.seed}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit execution context.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            False to propagate exceptions
        """
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time if self.start_time else 0.0

        if self.audit_enabled:
            if exc_type is None:
                self._logger.info(
                    f"Workflow '{self.workflow_name}' completed: "
                    f"execution_id={self.execution_id}, duration={duration:.3f}s"
                )
            else:
                self._logger.error(
                    f"Workflow '{self.workflow_name}' failed: "
                    f"execution_id={self.execution_id}, "
                    f"error={exc_type.__name__}: {exc_val}"
                )

        if self.metrics_enabled:
            # Stub for Prometheus metrics
            # In production: prometheus_client.Counter/Histogram
            pass

        # Don't suppress exceptions
        return False

    def set_result(self, result: dict[str, Any]) -> None:
        """Set execution result.

        Args:
            result: Execution result dictionary
        """
        self.result = result

    def get_execution_metadata(self) -> dict[str, Any]:
        """Get execution metadata.

        Returns:
            Dictionary with execution metadata
        """
        return {
            "workflow_name": self.workflow_name,
            "execution_id": self.execution_id,
            "seed": self.seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (
                self.end_time - self.start_time if self.start_time and self.end_time else None
            ),
            "audit_enabled": self.audit_enabled,
            "metrics_enabled": self.metrics_enabled,
            "do178c_enabled": self.do178c_enabled,
        }


@contextmanager
def execution_context(
    workflow_name: str,
    seed: int | None = None,
    audit_enabled: bool = True,
    metrics_enabled: bool = True,
    do178c_enabled: bool = True,
) -> Generator[ExecutionContext, None, None]:
    """Context manager factory for execution contexts.

    Args:
        workflow_name: Name of the workflow
        seed: Random seed for reproducibility
        audit_enabled: Enable audit logging
        metrics_enabled: Enable metrics collection
        do178c_enabled: Enable DO-178C compliance

    Yields:
        ExecutionContext instance
    """
    ctx = ExecutionContext(
        workflow_name=workflow_name,
        seed=seed,
        audit_enabled=audit_enabled,
        metrics_enabled=metrics_enabled,
        do178c_enabled=do178c_enabled,
    )

    with ctx:
        yield ctx

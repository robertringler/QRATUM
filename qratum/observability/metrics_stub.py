"""Prometheus metrics stub.

Stub for Prometheus metrics collection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Stub for Prometheus metrics collection.

    In production, this would use prometheus_client.
    """

    def __init__(self, enabled: bool = True):
        """Initialize metrics collector.

        Args:
            enabled: Enable metrics collection
        """
        self.enabled = enabled
        self._logger = logging.getLogger("qratum.metrics")

    def record_execution_time(self, workflow: str, duration: float) -> None:
        """Record workflow execution time.

        Args:
            workflow: Workflow name
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        self._logger.debug(f"Metrics: {workflow} executed in {duration:.3f}s")

    def record_backend_selection(self, backend: str) -> None:
        """Record backend selection.

        Args:
            backend: Backend name
        """
        if not self.enabled:
            return

        self._logger.debug(f"Metrics: Backend selected - {backend}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.

        Returns:
            Metrics dictionary
        """
        return {
            "enabled": self.enabled,
            "note": "Prometheus metrics stub - production uses prometheus_client",
        }

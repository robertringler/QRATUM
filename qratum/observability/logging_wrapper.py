"""Structured logging wrapper.

Wrapper for structured logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

__all__ = ["StructuredLogger"]


class StructuredLogger:
    """Wrapper for structured logging.

    Provides consistent structured logging across platform.
    """

    def __init__(self, name: str = "qratum", level: str = "INFO"):
        """Initialize structured logger.

        Args:
            name: Logger name
            level: Log level
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level))

    def log_workflow_start(self, workflow: str, metadata: Dict[str, Any]) -> None:
        """Log workflow start.

        Args:
            workflow: Workflow name
            metadata: Workflow metadata
        """
        self._logger.info(f"Workflow started: {workflow}", extra=metadata)

    def log_workflow_complete(
        self, workflow: str, duration: float, metadata: Dict[str, Any]
    ) -> None:
        """Log workflow completion.

        Args:
            workflow: Workflow name
            duration: Execution duration
            metadata: Workflow metadata
        """
        self._logger.info(f"Workflow completed: {workflow} ({duration:.3f}s)", extra=metadata)

    def log_error(self, message: str, error: Exception) -> None:
        """Log error.

        Args:
            message: Error message
            error: Exception
        """
        self._logger.error(f"{message}: {error}", exc_info=True)

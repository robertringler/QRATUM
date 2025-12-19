"""Audit logging wrapper.

Wraps existing audit functionality from quasim.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

__all__ = ["AuditWrapper"]


class AuditWrapper:
    """Wrapper for audit logging.

    Provides audit trail generation for DO-178C compliance.
    """

    def __init__(self, enabled: bool = True):
        """Initialize audit wrapper.

        Args:
            enabled: Enable audit logging
        """
        self.enabled = enabled
        self._logger = logging.getLogger("qratum.audit")
        self._audit_trail: list = []

    def log_start(
        self,
        workflow: str,
        execution_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log workflow start.

        Args:
            workflow: Workflow name
            execution_id: Execution identifier
            metadata: Optional metadata
        """
        if not self.enabled:
            return

        entry = {
            "event": "workflow_start",
            "workflow": workflow,
            "execution_id": execution_id,
            "metadata": metadata or {},
        }

        self._audit_trail.append(entry)
        self._logger.info(f"Audit: workflow_start - {workflow} ({execution_id})")

    def log_result(
        self,
        workflow: str,
        execution_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Log workflow result.

        Args:
            workflow: Workflow name
            execution_id: Execution identifier
            result: Result dictionary
        """
        if not self.enabled:
            return

        entry = {
            "event": "workflow_result",
            "workflow": workflow,
            "execution_id": execution_id,
            "result": result,
        }

        self._audit_trail.append(entry)
        self._logger.info(f"Audit: workflow_result - {workflow} ({execution_id})")

    def get_audit_trail(self) -> list:
        """Get audit trail.

        Returns:
            List of audit entries
        """
        return self._audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear audit trail."""
        self._audit_trail.clear()

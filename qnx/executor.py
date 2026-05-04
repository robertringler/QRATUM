"""Executor abstraction delegating to existing QNXSubstrate.

This module provides a high-level executor interface that delegates
to the existing QNXSubstrate implementation.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from typing import Any

from qnx.core import QNXSubstrate


class QNXExecutor:
    """High-level executor interface for quantum/classical execution.

    Delegates to the existing QNXSubstrate for actual task execution,
    providing a clean separation between executor API and substrate implementation.
    """

    def __init__(self) -> None:
        """Initialize executor with QNXSubstrate instance."""
        self.substrate = QNXSubstrate()

    def dispatch(self, task: Any) -> Any:
        """Dispatch a task for execution.

        Delegates to the substrate's run_simulation() method.

        Args:
            task: Task configuration to execute

        Returns:
            Result from substrate execution
        """
        return self.substrate.run_simulation(task)

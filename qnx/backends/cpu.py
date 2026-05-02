"""CPU backend stub for classical execution.

This module provides a stub for CPU-based execution.
Full implementation will be completed in future PRs.

Version: 1.0.0
Status: Production (Stub)
"""

from __future__ import annotations

from typing import Any


class CPUBackend:
    """CPU backend for classical computation.

    This is a stub implementation establishing the structural interface.
    Full CPU backend will be implemented in a future PR.
    """

    def run(self, task: Any) -> Any:
        """Execute task on CPU backend.

        This method will be implemented in PR-006 (Backend Implementation).

        Args:
            task: Task to execute

        Returns:
            Execution result

        Raises:
            NotImplementedError: Placeholder for PR-006
        """
        # GAP-STUB-006: CPU backend not yet active; return empty result dict
        return {}

"""Explicit orchestration layer delegating to existing kernel.

This module provides a high-level orchestration interface that delegates
to the existing QStackKernel implementation.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from typing import Any

from qstack.kernel import QStackKernel


class Orchestrator:
    """High-level orchestration layer for Q-Stack execution.
    
    Delegates to the existing QStackKernel for actual execution,
    providing a clean separation between orchestration and kernel logic.
    """
    
    def __init__(self, session: Any) -> None:
        """Initialize orchestrator with a system session.
        
        Args:
            session: SystemSession instance containing kernel and configuration
        """
        self.session = session
        self.kernel: QStackKernel = session.kernel
    
    def execute(self) -> Any:
        """Execute the orchestrated workflow.
        
        Delegates to the kernel's boot() method for actual execution.
        
        Returns:
            Result from kernel boot operation
        """
        return self.kernel.boot()

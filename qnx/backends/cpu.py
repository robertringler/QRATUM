"""CPU backend stub for classical execution.

This module provides a stub for CPU-based execution.

IMPORTANT: This is a STUB IMPLEMENTATION. For production use, use the
reference implementations in quasim/opt/ultra_sssp.py which provide
verified correctness against Dijkstra baseline.

Version: 1.0.0
Status: STUB - Reference implementation available elsewhere
Roadmap: PR-006 (Backend Implementation)

AUDIT NOTE: This stub exists to establish the interface contract.
For correctness-verified computation, use:
- quasim.opt.ultra_sssp.dijkstra_baseline (reference)
- quasim.opt.post_dijkstra_sssp.PostDijkstraSSSP (accelerated)
"""

from __future__ import annotations

import warnings
from typing import Any


class CPUBackendStub(NotImplementedError):
    """Exception raised when CPU backend stub method is called."""
    pass


class CPUBackend:
    """CPU backend for classical computation.
    
    WARNING: This is a STUB implementation establishing the interface.
    For production use, use reference implementations in quasim.opt module.
    
    The reference implementations provide:
    - Verified correctness against Dijkstra baseline
    - Deterministic execution
    - Full test coverage
    
    Attributes:
        available: Always False for stub (use reference implementations)
        
    Example:
        >>> # Instead of using this stub, use:
        >>> from quasim.opt.ultra_sssp import dijkstra_baseline
        >>> distances, metrics = dijkstra_baseline(graph, source=0)

    This is a stub implementation establishing the structural interface.
    Full CPU backend will be implemented in a future PR.
    """

    _warned = False

    def __init__(self):
        """Initialize CPU backend stub with guidance warning."""
        if not CPUBackend._warned:
            warnings.warn(
                "CPUBackend is a stub implementation. "
                "For production, use reference implementations in quasim.opt module: "
                "dijkstra_baseline() or PostDijkstraSSSP.solve()",
                UserWarning,
                stacklevel=2
            )
            CPUBackend._warned = True

    @property
    def available(self) -> bool:
        """Check if CPU backend is available.
        
        Returns:
            False - use reference implementations instead
        """
        return False

    def is_available(self) -> bool:
        """Check if CPU backend is available.
        
        Returns:
            False - use reference implementations instead
        """
        return False

    def run(self, task: Any) -> Any:
        """Execute task on CPU backend.
        
        WARNING: This method always raises CPUBackendStub.
        Use reference implementations instead.
        

        This method will be implemented in PR-006 (Backend Implementation).

        Args:
            task: Task to execute

        Returns:
            Never returns - always raises exception
            
            Execution result

        Raises:
            CPUBackendStub: Stub method called; use reference implementations
        """
        raise CPUBackendStub(
            "CPUBackend.run() is a stub. "
            "Use reference implementations for production: "
            "quasim.opt.ultra_sssp.dijkstra_baseline() or "
            "quasim.opt.post_dijkstra_sssp.PostDijkstraSSSP.solve(). "
            "Full backend planned for PR-006."
        )
        # GAP-STUB-006: CPU backend not yet active; return empty result dict
        return {}

"""GPU backend stub for accelerated execution.

This module provides a stub for GPU-based execution.

IMPORTANT: This is a STUB IMPLEMENTATION. No GPU acceleration is currently
available. All calls to GPUBackend.run() will raise NotImplementedError.

For production use, fall back to CPUBackend or use the reference
implementation in quasim/opt/ultra_sssp.py.

Version: 1.0.0
Status: STUB - NOT IMPLEMENTED
Roadmap: PR-006 (Backend Implementation)

AUDIT NOTE: This stub exists to establish the interface contract.
Actual GPU implementation requires CUDA/OpenCL integration.
"""

from __future__ import annotations

import warnings
from typing import Any


class GPUBackendNotAvailable(NotImplementedError):
    """Exception raised when GPU backend is not available."""
    pass


class GPUBackend:
    """GPU backend for accelerated computation.
    
    WARNING: This is a STUB implementation. No GPU acceleration is available.
    All method calls will raise GPUBackendNotAvailable.
    
    For correctness-verified computation, use the reference CPU implementation.
    
    Attributes:
        available: Always False for stub implementation
        
    Example:
        >>> backend = GPUBackend()
        >>> backend.is_available()
        False
        >>> backend.run(task)  # Raises GPUBackendNotAvailable

    This is a stub implementation establishing the structural interface.
    Full GPU backend will be implemented in a future PR.
    """

    _warned = False

    def __init__(self):
        """Initialize GPU backend stub with deprecation warning."""
        if not GPUBackend._warned:
            warnings.warn(
                "GPUBackend is a stub implementation. "
                "GPU acceleration is NOT available. "
                "Use CPUBackend or reference implementations for production.",
                UserWarning,
                stacklevel=2
            )
            GPUBackend._warned = True

    @property
    def available(self) -> bool:
        """Check if GPU backend is available.
        
        Returns:
            Always False for stub implementation
        """
        return False

    def is_available(self) -> bool:
        """Check if GPU backend is available.
        
        Returns:
            Always False for stub implementation
        """
        return False

    def run(self, task: Any) -> Any:
        """Execute task on GPU backend.
        
        WARNING: This method always raises GPUBackendNotAvailable.
        

        This method will be implemented in PR-006 (Backend Implementation).

        Args:
            task: Task to execute

        Returns:
            Never returns - always raises exception
            
            Execution result

        Raises:
            GPUBackendNotAvailable: GPU backend is not implemented
        """
        raise GPUBackendNotAvailable(
            "GPU backend is NOT implemented. "
            "This is a stub establishing the interface contract. "
            "Use CPUBackend or reference implementations for production. "
            "GPU support planned for PR-006 (Backend Implementation)."
        )
        # GAP-STUB-007: GPU backend not yet active; return empty result dict
        return {}

"""Quantum backend integration adapters.

Adapters for integrating quasim.quantum modules with platform layer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["QuantumBackendAdapter"]


class QuantumBackendAdapter:
    """Adapter for quantum backend operations.

    Wraps quasim.quantum modules without modifying them.
    """

    def __init__(
        self,
        backend_type: str = "simulator",
        seed: Optional[int] = None,
        shots: int = 1024,
    ):
        """Initialize quantum backend adapter.

        Args:
            backend_type: Backend type ('simulator', 'ibmq', 'cuquantum')
            seed: Random seed for reproducibility
            shots: Number of shots
        """
        self.backend_type = backend_type
        self.seed = seed
        self.shots = shots

    def get_config(self) -> Dict[str, Any]:
        """Get backend configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "backend_type": self.backend_type,
            "seed": self.seed,
            "shots": self.shots,
        }

    def validate_backend(self) -> bool:
        """Validate backend is available.

        Returns:
            True if backend is available
        """
        # Stub implementation - in production, check actual backend availability
        return True

"""Seed manager wrapper for NIST SP 800-90A compliance.

Wraps existing seed_management module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Import existing seed manager
seed_mgmt_path = Path(__file__).parent.parent.parent / "seed_management"
sys.path.insert(0, str(seed_mgmt_path))

try:
    from seed_manager import SeedManager as _SeedManager
except ImportError:
    _SeedManager = None

__all__ = ["SeedManagerWrapper"]


class SeedManagerWrapper:
    """Wrapper for seed management with NIST SP 800-90A compliance.

    Wraps existing seed_management.SeedManager.
    """

    def __init__(self, base_seed: int = 42, environment: str = "default"):
        """Initialize seed manager wrapper.

        Args:
            base_seed: Base seed for deterministic generation
            environment: Environment identifier
        """
        self.base_seed = base_seed
        self.environment = environment

        # Use existing SeedManager if available
        if _SeedManager:
            self._manager = _SeedManager(base_seed=base_seed, environment=environment)
        else:
            self._manager = None

    def get_execution_seed(self, replay_id: Optional[str] = None) -> int:
        """Get execution seed.

        Args:
            replay_id: Optional replay identifier

        Returns:
            Seed value
        """
        if self._manager:
            record = self._manager.generate_seed(replay_id)
            return record.seed_value
        else:
            # Fallback if seed manager not available
            return self.base_seed

    def export_manifest(self) -> dict:
        """Export seed manifest for audit trail.

        Returns:
            Manifest dictionary
        """
        if self._manager:
            return self._manager.export_manifest()
        else:
            return {
                "base_seed": self.base_seed,
                "environment": self.environment,
                "note": "Seed manager module not available",
            }

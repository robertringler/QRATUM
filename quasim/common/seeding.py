"""Deterministic seeding utilities for reproducible simulations.

Provides global seed management and configuration hashing.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set global random seed for all RNG sources.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    # Try to set torch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hash_config(config: dict[str, Any]) -> str:
    """Generate deterministic hash of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hex digest of config
    """
    # Sort keys for determinism
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def derive_seed(base_seed: int, suffix: str) -> int:
    """Derive a child seed from a base seed and suffix.

    Args:
        base_seed: Base random seed
        suffix: String suffix to differentiate derived seeds

    Returns:
        Derived seed value
    """
    combined = f"{base_seed}:{suffix}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big")


class SeedManager:
    """Manager for deterministic seed generation.

    Attributes:
        base_seed: Base random seed
        derived_seeds: Cache of derived seeds
    """

    def __init__(self, base_seed: int = 42):
        """Initialize seed manager.

        Args:
            base_seed: Base random seed
        """
        self.base_seed = base_seed
        self.derived_seeds: dict[str, int] = {}
        set_global_seed(base_seed)

    def get_seed(self, name: str) -> int:
        """Get or create a derived seed for a named component.

        Args:
            name: Component name

        Returns:
            Derived seed value
        """
        if name not in self.derived_seeds:
            self.derived_seeds[name] = derive_seed(self.base_seed, name)
        return self.derived_seeds[name]

    def reset(self, base_seed: Optional[int] = None) -> None:
        """Reset seed manager.

        Args:
            base_seed: New base seed (if None, keeps current)
        """
        if base_seed is not None:
            self.base_seed = base_seed
        self.derived_seeds.clear()
        set_global_seed(self.base_seed)

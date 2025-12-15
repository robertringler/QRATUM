"""QRATUM global configuration system.

Provides centralized configuration management with support for
.qratum/ directory, environment variables, and runtime overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class QRATUMConfig:
    """Global QRATUM configuration.

    Attributes:
        backend: Default backend ('cpu', 'gpu', 'multi-gpu', 'tensor-network', 'stabilizer')
        precision: Floating point precision ('fp8', 'fp16', 'fp32', 'fp64')
        max_qubits: Maximum number of qubits (for validation)
        seed: Random seed for reproducibility
        cache_dir: Directory for caching compiled circuits
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        telemetry_enabled: Enable telemetry collection
    """

    backend: str = "cpu"
    precision: str = "fp32"
    max_qubits: int = 40
    seed: Optional[int] = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".qratum" / "cache")
    log_level: str = "INFO"
    telemetry_enabled: bool = False

    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Override from environment variables
        if env_backend := os.getenv("QRATUM_BACKEND"):
            self.backend = env_backend
        if env_precision := os.getenv("QRATUM_PRECISION"):
            self.precision = env_precision
        if env_seed := os.getenv("QRATUM_SEED"):
            self.seed = int(env_seed)
        if env_log := os.getenv("QRATUM_LOG_LEVEL"):
            self.log_level = env_log

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "backend": self.backend,
            "precision": self.precision,
            "max_qubits": self.max_qubits,
            "seed": self.seed,
            "cache_dir": str(self.cache_dir),
            "log_level": self.log_level,
            "telemetry_enabled": self.telemetry_enabled,
        }


# Global configuration instance
_global_config: Optional[QRATUMConfig] = None


def get_config() -> QRATUMConfig:
    """Get global QRATUM configuration.

    Returns:
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = QRATUMConfig()
    return _global_config


def set_config(config: QRATUMConfig) -> None:
    """Set global QRATUM configuration.

    Args:
        config: New configuration instance
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to defaults."""
    global _global_config
    _global_config = QRATUMConfig()


__all__ = [
    "QRATUMConfig",
    "get_config",
    "set_config",
    "reset_config",
]

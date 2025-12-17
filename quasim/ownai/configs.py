"""Configuration management for QuASIM-Own AI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class QuasimOwnConfig:
    """Central configuration for QuASIM-Own AI.

    Attributes
    ----------
    seed : int
        Global random seed for determinism (default: 1337)
    mixed_precision : bool
        Enable AMP (Automatic Mixed Precision) for training (default: False)
    cache_dir : Path
        Directory for caching datasets and models (default: .cache/quasim_datasets/)
    output_dir : Path
        Directory for run outputs and logs (default: runs/)
    precision : Literal["fp32", "fp16", "bf16"]
        Floating point precision mode (default: "fp32")
    cpu_only : bool
        Force CPU-only execution (default: True)
    deterministic : bool
        Enable deterministic PyTorch operations (default: True)
    """

    seed: int = 1337
    mixed_precision: bool = False
    cache_dir: Path = Path(".cache/quasim_datasets")
    output_dir: Path = Path("runs")
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    cpu_only: bool = True
    deterministic: bool = True

    def __post_init__(self):
        """Ensure directories exist."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global default configuration
DEFAULT_CONFIG = QuasimOwnConfig()

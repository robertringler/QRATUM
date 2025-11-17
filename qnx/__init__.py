"""QNX substrate package for orchestrating QuASIM simulations.

This package provides a pluggable substrate that can dispatch simulations to
multiple backends (modern, legacy, and QVR) while producing a normalized result
record.
"""

from .core import QNXSubstrate
from .types import SimulationConfig, SubstrateResult, SecurityLevel

__all__ = [
    "QNXSubstrate",
    "SimulationConfig",
    "SubstrateResult",
    "SecurityLevel",
]

__version__ = "3.0.0"

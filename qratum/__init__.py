"""
QRATUM Sovereign AI Platform v2.0

A production-grade platform for deterministic, auditable AI computations
across 14 vertical domains with full QRATUM invariant enforcement.

Core Modules (per QRATUM ASCENSION DIRECTIVE):
- metrics: QRATUM metrics (OSR, CEI, SF, HRD)
- discovery: Self-Discovery Cycle (SDC) implementation
- sovereign_stack: Sovereign Stack layer formalization
"""

__version__ = "2.0.0"
__legacy_name__ = "QuASIM"
__license__ = "Apache-2.0"
__url__ = "https://qratum.io"
__github__ = "https://github.com/robertringler/QRATUM"

# Quantum core (numpy state-vector simulator in qratum.core).
from .core import (
    Circuit,
    DensityMatrix,
    Measurement,
    Result,
    Simulator,
    StateVector,
    gates,
)

# Platform classes pull in the compliance/observability/opt/quantum/workflows
# stack, so resolve them lazily (PEP 562) to keep plain `import qratum` light.
_LAZY_PLATFORM_EXPORTS = {
    "QRATUMPlatform": ("qratum.core.platform", "QRATUMPlatform"),
    "create_platform": ("qratum.core.platform", "create_platform"),
    "PlatformConfig": ("qratum.core.platform_config", "PlatformConfig"),
    "QRATUMConfig": ("qratum.config", "QRATUMConfig"),
    "get_config": ("qratum.config", "get_config"),
    "set_config": ("qratum.config", "set_config"),
    "reset_config": ("qratum.config", "reset_config"),
}


def __getattr__(name):
    if name in _LAZY_PLATFORM_EXPORTS:
        import importlib

        module_name, attr = _LAZY_PLATFORM_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "platform",
    "verticals",
    # Core directive modules
    "metrics",
    "discovery",
    "sovereign_stack",
    # Quantum core
    "Circuit",
    "Simulator",
    "StateVector",
    "Measurement",
    "Result",
    "DensityMatrix",
    "gates",
    # Platform (lazy)
    "QRATUMPlatform",
    "create_platform",
    "PlatformConfig",
    "QRATUMConfig",
    "get_config",
    "set_config",
    "reset_config",
]

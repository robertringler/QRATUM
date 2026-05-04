"""QRATUM Sovereign AI Platform v2.0 - Core Infrastructure.

This module provides the foundational classes and types for the QRATUM platform,
including vertical modules, compute substrates, contracts, and event chains.
"""

from qratum_platform.core import (
    ComputeSubstrate,
    ExecutionEvent,
    MerkleEventChain,
    PlatformContract,
    PlatformIntent,
    QRATUMPlatform,
    SafetyViolation,
    VerticalModule,
    VerticalModuleBase,
)
from qratum_platform.substrates import VERTICAL_SUBSTRATE_MAPPINGS

__all__ = [
    "VerticalModule",
    "ComputeSubstrate",
    "PlatformIntent",
    "PlatformContract",
    "ExecutionEvent",
    "MerkleEventChain",
    "VerticalModuleBase",
    "QRATUMPlatform",
    "SafetyViolation",
    "VERTICAL_SUBSTRATE_MAPPINGS",
]

__version__ = "2.0.0"

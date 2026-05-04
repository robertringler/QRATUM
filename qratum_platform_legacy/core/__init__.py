"""QRATUM Platform Core Infrastructure.

Core components for the QRATUM Sovereign AI Platform:
- Intent and contract system
- Event chain for cryptographic audit
- Base classes for vertical modules
- Platform orchestrator
- Compute substrate mappings
"""

from platform.core.base import VerticalModuleBase
from platform.core.events import ExecutionEvent, MerkleEventChain
from platform.core.intent import PlatformContract, PlatformIntent
from platform.core.orchestrator import QRATUMPlatform
from platform.core.substrates import ComputeSubstrate

__all__ = [
    "PlatformIntent",
    "PlatformContract",
    "ExecutionEvent",
    "MerkleEventChain",
    "VerticalModuleBase",
    "QRATUMPlatform",
    "ComputeSubstrate",
]

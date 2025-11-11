"""QuNimbus — Quantum-Optimized Cloud Fabric for QuASIM.

This module provides the orchestration layer for QuNimbus Wave 3,
including pilot generation, China Photonic Factory integration,
and cross-border quantum networking.
"""

__version__ = "2.0.0"
__wave__ = "3"

from quasim.qunimbus.china_integration import ChinaPhotonicFactory
from quasim.qunimbus.orchestrator import QuNimbusOrchestrator
from quasim.qunimbus.pilot_factory import PilotFactory

__all__ = [
    "QuNimbusOrchestrator",
    "PilotFactory",
    "ChinaPhotonicFactory",
]

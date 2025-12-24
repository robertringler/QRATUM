"""AION Concurrency Module.

Implements the formal concurrency lattice for effect tracking:
- Bottom (pure) ↑ ThreadSpawn/Join ↑ ChannelSend/Recv ↑ ActorSend ↑ WarpSync ↑ PipelineStage ↑ Top

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .lattice import (
    ConcurrencyEffect,
    EffectLattice,
    EffectCapability,
    EffectChecker,
    FunctionEffect,
)

__all__ = [
    "ConcurrencyEffect",
    "EffectLattice",
    "EffectCapability",
    "EffectChecker",
    "FunctionEffect",
]

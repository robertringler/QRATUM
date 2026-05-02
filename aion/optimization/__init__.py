"""AION Optimization Module.

Implements optimization and fusion for AION-SIR:
- Cross-language kernel fusion
- Proof-preserving optimizations
- Adaptive scheduling

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .fusion import (
    CrossLanguageFuser,
    FusionPattern,
    KernelFusion,
    detect_fusion_patterns,
)
from .scheduler import (
    AdaptiveScheduler,
    CausalScheduler,
    ScheduleResult,
    Task,
)

__all__ = [
    "KernelFusion",
    "FusionPattern",
    "CrossLanguageFuser",
    "detect_fusion_patterns",
    "AdaptiveScheduler",
    "CausalScheduler",
    "Task",
    "ScheduleResult",
]

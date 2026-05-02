"""Q-Stack core integration facade.

This package exposes deterministic adapters and orchestration utilities that
wrap the QNX substrate, QuASIM runtime, and QuNimbus governance layers behind a
single, cohesive API surface.
"""

from qstack.config import QStackConfig
from qstack.system import QStackSystem

__all__ = ["QStackConfig", "QStackSystem"]

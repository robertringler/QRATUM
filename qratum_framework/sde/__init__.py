"""qratum_framework.sde — Streaming Drift Engine (Module I).

Real-time, deterministic drift detector that runs *orthogonally* to
the canonical 9-step loop.  The SDE observes the same token / state
stream the domain executor produces and emits real-time verdicts
about constraint-boundary proximity *during* execution — not after.

Key invariant: the SDE has exactly **one** actuator privilege — it
may inject a synthetic HALT intent into the RIC stage via the
``halt_callback`` supplied to :class:`StreamingDriftEngine`.  It
never silently swallows exceptions; it never mutates pipeline state
directly.

Public API
----------

* :class:`Token`, :class:`WindowSnapshot` — fundamental units
* :class:`RollingWindowBuffer` — bounded deque with a min-population gate
* :class:`DriftScorer` (Protocol) and :class:`DriftReading`
* :class:`KLDriftScorer`, :class:`MMDDriftScorer` — concrete scorers
* :class:`AlarmEvaluator`, :class:`AlarmTier`, :class:`AlarmThresholds`,
  :class:`DriftEvent` — alarm-tier state machine
* :class:`StreamingDriftEngine`, :class:`SDESnapshot`, :data:`HaltCallback`
  — top-level orchestrator
* :class:`SDEConfig` — config sub-block extending
  :class:`qratum_framework.config.PipelineProfile`
"""

from qratum_framework.sde.buffer import RollingWindowBuffer
from qratum_framework.sde.config import SDEConfig
from qratum_framework.sde.engine import (
    HaltCallback,
    SDESnapshot,
    StreamingDriftEngine,
)
from qratum_framework.sde.evaluator import (
    AlarmEvaluator,
    AlarmThresholds,
    AlarmTier,
    DriftEvent,
)
from qratum_framework.sde.scorers import (
    DEGENERATE,
    DriftReading,
    DriftScorer,
    KLDriftScorer,
    MMDDriftScorer,
)
from qratum_framework.sde.tokens import Token, WindowSnapshot

__all__ = [
    # tokens
    "Token",
    "WindowSnapshot",
    # buffer
    "RollingWindowBuffer",
    # scorers
    "DEGENERATE",
    "DriftReading",
    "DriftScorer",
    "KLDriftScorer",
    "MMDDriftScorer",
    # evaluator
    "AlarmTier",
    "AlarmThresholds",
    "DriftEvent",
    "AlarmEvaluator",
    # engine
    "HaltCallback",
    "SDESnapshot",
    "StreamingDriftEngine",
    # config
    "SDEConfig",
]

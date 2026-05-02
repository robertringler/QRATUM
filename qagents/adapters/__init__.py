"""Reality Interface Controller (RIC) adapters.

This subpackage wires the deterministic
:class:`qagents.reality_interface.RealityInterfaceController` to the three
top-level QRATUM subsystems:

* :mod:`qagents.adapters.qratum` — QRATUM Sovereign-Stack metrics
  (OSR / CEI / SF / HRD) per ``qratum/metrics.py``.
* :mod:`qagents.adapters.ciir`  — CIIR variational dynamics (loss,
  constraint violation, entropy regulariser) per ``quasim/ciir``.
* :mod:`qagents.adapters.crs`   — Constrained Recursive Self-improvement
  (immutable boundaries + prohibited goals) per
  ``qratum_asi/orchestrator_master.py``.

Each adapter exposes a :class:`~qagents.reality_interface.Simulator`,
a :class:`~qagents.reality_interface.Proposer`, and a
``make_<system>_controller`` convenience that returns a fully-wired
:class:`~qagents.reality_interface.RealityInterfaceController`.

Adapters are pure-Python and operate on the canonical world-state schema
of each subsystem (no heavy imports from ``quasim``/``qratum_asi``), so
they remain importable in minimal environments and can run alongside —
or in place of — the real engines.
"""

from qagents.adapters.ciir import (
    ciir_proposer,
    ciir_simulator,
    make_ciir_controller,
)
from qagents.adapters.crs import (
    IMMUTABLE_BOUNDARIES,
    PROHIBITED_GOALS,
    crs_proposer,
    crs_simulator,
    make_crs_controller,
)
from qagents.adapters.qratum import (
    make_qratum_controller,
    qratum_proposer,
    qratum_simulator,
)

__all__ = [
    "make_qratum_controller",
    "qratum_simulator",
    "qratum_proposer",
    "make_ciir_controller",
    "ciir_simulator",
    "ciir_proposer",
    "make_crs_controller",
    "crs_simulator",
    "crs_proposer",
    "IMMUTABLE_BOUNDARIES",
    "PROHIBITED_GOALS",
]

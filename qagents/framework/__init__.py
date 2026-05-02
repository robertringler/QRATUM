"""CIIR–CRS–RIC constraint-governed closed-loop execution framework.

Three strictly separated modules:

    ciir  — governing theory (constraint algebra, observer map, satisfaction)
    crs   — computational substrate (state space, transition function T_C)
    ric   — deterministic controller (intent parser π, admissible set A_C, ρ)
"""

from qagents.framework.ciir import (Constraint, ConstraintAlgebra, ObserverMap,
                                    State)
from qagents.framework.crs import CRS, Action, FailureMode
from qagents.framework.ric import RIC, Intent, TraceEntry

__all__ = [
    "Constraint",
    "ConstraintAlgebra",
    "State",
    "ObserverMap",
    "CRS",
    "Action",
    "FailureMode",
    "RIC",
    "Intent",
    "TraceEntry",
]

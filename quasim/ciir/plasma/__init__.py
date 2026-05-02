r"""CIIR Plasma Reconnection Control Engine.

A 2D resistive-MHD scaffold + control loop + CIIR (Constraints / Interfaces /
Recursion) lens for studying magnetic reconnection control.

**Physics labels** (in source docstrings):
    [V] = established physics, [D] = derived from established results,
    [H] = explicit hypothesis. CIIR is a structuring lens, not physics.

Quick start
-----------
>>> from quasim.ciir.plasma import EngineConfig, run_reconnection_control
>>> cfg = EngineConfig(nx=32, ny=32, n_steps=10)
>>> result = run_reconnection_control(cfg)
>>> snap = result.ciir_history[-1]
>>> snap.topology.n_X >= 0
True

Modules
-------
mhd       : 2D reduced resistive MHD state, time stepper.
topology  : Interface manifold 𝕀: X/O-points, current sheets.
spectrum  : Instability spectrum 𝕀(k, t), shell spectra, growth rates.
control   : Control vector u, objective functional, ReconnectionController.
ciir      : (C, I, R) lens, CIIRSnapshot, stability inequality, failure modes.
engine    : End-to-end EngineConfig + run_reconnection_control().
"""

from __future__ import annotations

from .ciir import (FAILURE_MODES, CIIRSnapshot, Constraints, FailureMode,
                   Interfaces, Recursion, make_snapshot, stability_inequality)
from .control import (ControllerConfig, ControlVector, ObjectiveValue,
                      ObjectiveWeights, ReconnectionController,
                      energy_release_rate, evaluate_objective,
                      gaussian_target_mask, plasmoid_indicator,
                      reconnection_localization, zero_control_vector)
from .engine import (EngineConfig, RunResult, initial_state,
                     run_reconnection_control)
from .mhd import (Grid2D, MHDState, harris_sheet, perturb_tearing,
                  recommended_dt, rhs, step_rk2, zero_control)
from .spectrum import (SpectrumSnapshot, fkr_growth_rate_estimate,
                       growth_rates, lundquist_number,
                       plasmoid_growth_rate_estimate, spectrum)
from .topology import (CriticalPoint, CurrentSheet, TopologySnapshot,
                       find_critical_points, find_current_sheets,
                       reconnection_rate_field, topology_snapshot)

__all__ = [
    # mhd
    "Grid2D",
    "MHDState",
    "harris_sheet",
    "perturb_tearing",
    "recommended_dt",
    "rhs",
    "step_rk2",
    "zero_control",
    # topology
    "CriticalPoint",
    "CurrentSheet",
    "TopologySnapshot",
    "find_critical_points",
    "find_current_sheets",
    "reconnection_rate_field",
    "topology_snapshot",
    # spectrum
    "SpectrumSnapshot",
    "spectrum",
    "growth_rates",
    "lundquist_number",
    "fkr_growth_rate_estimate",
    "plasmoid_growth_rate_estimate",
    # control
    "ControllerConfig",
    "ControlVector",
    "ObjectiveValue",
    "ObjectiveWeights",
    "ReconnectionController",
    "evaluate_objective",
    "energy_release_rate",
    "gaussian_target_mask",
    "plasmoid_indicator",
    "reconnection_localization",
    "zero_control_vector",
    # ciir
    "CIIRSnapshot",
    "Constraints",
    "FailureMode",
    "FAILURE_MODES",
    "Interfaces",
    "Recursion",
    "make_snapshot",
    "stability_inequality",
    # engine
    "EngineConfig",
    "RunResult",
    "initial_state",
    "run_reconnection_control",
]

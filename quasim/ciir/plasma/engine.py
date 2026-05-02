r"""End-to-end CIIR Reconnection Control Loop (problem statement §6).

This is the orchestrator that wires together:

1. Initialization (Harris sheet equilibrium + tearing-seed perturbation).
2. Interface detection (X-points, current sheets) — :mod:`topology`.
3. Instability spectrum analysis — :mod:`spectrum`.
4. Control synthesis — :mod:`control`.
5. Actuation (E_drive, η_eff, F_ω applied through the RHS hook).
6. Recursion update (RK2 step of resistive-MHD equations).

Output trajectory
-----------------
:class:`RunResult` carries:

- ``state_history`` — every k-th MHD state (memory-bounded).
- ``ciir_history`` — :class:`CIIRSnapshot` per recorded step.
- ``objective_history`` — :class:`ObjectiveValue` per recorded step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .ciir import CIIRSnapshot, make_snapshot
from .control import (ControllerConfig, ObjectiveValue, ObjectiveWeights,
                      ReconnectionController, evaluate_objective,
                      gaussian_target_mask, zero_control_vector)
from .mhd import (Grid2D, MHDState, harris_sheet, perturb_tearing,
                  recommended_dt, step_rk2)

FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Configuration for :func:`run_reconnection_control`."""

    nx: int = 64
    ny: int = 64
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    eta: float = 5.0e-3
    nu: float = 5.0e-3
    Bz: float = 1.0
    sheet_amplitude: float = 1.0  # B0 in Harris sheet
    sheet_width: float = 0.5  # 'a' in Harris sheet
    perturbation_amplitude: float = 1.0e-3
    perturbation_mode: int = 1
    n_steps: int = 50
    dt: Optional[float] = None  # if None, auto from recommended_dt()
    record_every: int = 1
    target_x: Optional[float] = None  # default: lower-sheet center
    target_y: Optional[float] = None
    target_sigma: float = 0.4
    seed: int = 0
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    use_controller: bool = True


@dataclass
class RunResult:
    """Result of an end-to-end run."""

    config: EngineConfig
    times: list[float] = field(default_factory=list)
    state_history: list[MHDState] = field(default_factory=list)
    ciir_history: list[CIIRSnapshot] = field(default_factory=list)
    objective_history: list[ObjectiveValue] = field(default_factory=list)
    divB_max: list[float] = field(default_factory=list)
    energy_total: list[float] = field(default_factory=list)
    final_state: Optional[MHDState] = None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initial_state(cfg: EngineConfig) -> MHDState:
    """Construct a Harris-sheet equilibrium with a tearing-seed perturbation [V]."""
    grid = Grid2D(nx=cfg.nx, ny=cfg.ny, Lx=cfg.Lx, Ly=cfg.Ly)
    rng = np.random.default_rng(cfg.seed)
    psi0 = harris_sheet(grid, B0=cfg.sheet_amplitude, a=cfg.sheet_width)
    delta = perturb_tearing(
        grid,
        amplitude=cfg.perturbation_amplitude,
        mode=cfg.perturbation_mode,
        rng=rng,
    )
    psi = psi0 + delta
    omega = np.zeros_like(psi)
    return MHDState(psi=psi, omega=omega, grid=grid, eta=cfg.eta, nu=cfg.nu, Bz=cfg.Bz)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_reconnection_control(cfg: EngineConfig) -> RunResult:
    """Execute the CIIR reconnection control loop.

    The loop follows the 6-step algorithm in the problem statement:
    (1) initialize → (2) interface detect → (3) spectrum → (4) control synthesis
    → (5) actuate (RHS forcings) → (6) recursion (RK2 step). Repeat.

    All [V] physics lives inside :func:`mhd.step_rk2`. The controller and CIIR
    snapshots live entirely in the [D]/interpretive layer.
    """
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if cfg.record_every <= 0:
        raise ValueError("record_every must be positive.")

    state = initial_state(cfg)
    grid = state.grid

    # Default target: lower Harris sheet midpoint
    tx = cfg.target_x if cfg.target_x is not None else grid.Lx / 2.0
    ty = cfg.target_y if cfg.target_y is not None else grid.Ly / 4.0
    target = gaussian_target_mask(state, tx, ty, cfg.target_sigma)

    controller = ReconnectionController(cfg.controller_config) if cfg.use_controller else None

    if cfg.dt is None:
        # Use rough V_A ~ B0 as characteristic speed.
        dt = recommended_dt(grid, eta=cfg.eta, nu=cfg.nu, B_max=cfg.sheet_amplitude)
    else:
        dt = float(cfg.dt)

    result = RunResult(config=cfg)

    def _record(step_idx: int, control_norm: float, obj: ObjectiveValue) -> None:
        snap = make_snapshot(state, control_norm=control_norm)
        result.times.append(state.t)
        result.state_history.append(state.copy())
        result.ciir_history.append(snap)
        result.objective_history.append(obj)
        result.divB_max.append(state.divB_max())
        result.energy_total.append(state.magnetic_energy() + state.kinetic_energy())

    # Initial recording (step 0).
    obj0 = evaluate_objective(state, cfg.weights, target, k_band=cfg.controller_config.k_band)
    _record(0, control_norm=0.0, obj=obj0)

    for step in range(1, cfg.n_steps + 1):
        # (4) Control synthesis
        if controller is not None:
            ctrl_vec = controller.synthesize(state, target)
        else:
            ctrl_vec = zero_control_vector(state)
        hook = ctrl_vec.as_hook()

        # (5)+(6) Actuation + recursion
        step_rk2(state, dt, control=hook)

        # Diagnostics + objective
        obj = evaluate_objective(state, cfg.weights, target, k_band=cfg.controller_config.k_band)

        if step % cfg.record_every == 0 or step == cfg.n_steps:
            # Use the objective total as a coarse control_norm proxy [D].
            _record(step, control_norm=obj.total, obj=obj)

    result.final_state = state.copy()
    return result


__all__ = [
    "EngineConfig",
    "RunResult",
    "initial_state",
    "run_reconnection_control",
]

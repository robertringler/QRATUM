r"""Control vector and reconnection-shaping controller.

[V] = established physics. [D] = derived. [H] = hypothesis.
The control synthesis is **interpretive** (CIIR lens); only the underlying
PDE forcings (E_drive, η, ν, vorticity sources) are physical.

Control vector (problem-statement §2)
-------------------------------------
.. math::
    u(t) = \{ B_{ext}(x, t),\; E_{drive}(x, t),\; \eta_{eff}(x, t),\; p_{drive}(x, t) \}

In the 2D RMHD scaffold these enter as:

- ``E_drive``           → adds to ``∂_t ψ`` directly (induction forcing).
- ``eta_eff``           → replaces the constant ``η`` in
  :math:`\nabla\!\cdot(\eta_{eff}\nabla\psi)`.
- ``B_ext`` (via ``δψ_ext``) → an additive boundary/equilibrium flux applied
  through ``E_drive = ∂_t δψ_ext``. We expose it as a static flux template
  whose time derivative the controller injects.
- ``p_drive``           → a vorticity source ``F_ω = (∂_x ∂_y - ∂_y ∂_x) p ≡ 0``
  in scalar form; in RMHD a pressure-driven baroclinic torque is
  :math:`F_\omega = \hat z \cdot (\nabla \rho \times \nabla p)/\rho^2` [V]. For
  the scalar incompressible scaffold we model ``p_drive`` as a direct vorticity
  source field (i.e. the user supplies :math:`F_\omega` directly) — this is
  flagged as a modeling choice [D], not first-principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .mhd import ControlHook, MHDState

FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Control vector
# ---------------------------------------------------------------------------


@dataclass
class ControlVector:
    """Time-dependent control fields.

    Each field has the same shape as the grid. ``E_drive``, ``eta_eff``, and
    ``F_omega`` enter the RHS directly (see :mod:`mhd`). ``psi_ext`` is a
    static external flux template; its time derivative is added to ``E_drive``
    via :meth:`as_hook`.
    """

    E_drive: FloatArray
    eta_eff: FloatArray
    F_omega: FloatArray
    psi_ext: Optional[FloatArray] = None
    psi_ext_rate: float = 0.0

    def as_hook(self) -> ControlHook:
        """Return a :class:`ControlHook` callable for use with :func:`mhd.step_rk2`."""
        E0 = self.E_drive
        eta0 = self.eta_eff
        F0 = self.F_omega
        psi_ext = self.psi_ext
        rate = float(self.psi_ext_rate)

        def hook(state: MHDState, t: float) -> tuple[FloatArray, FloatArray, FloatArray]:
            E = E0 if psi_ext is None else (E0 + rate * psi_ext)
            return E, eta0, F0

        return hook


def zero_control_vector(state: MHDState) -> ControlVector:
    """Construct a no-op control vector with the right shapes for ``state``."""
    z = np.zeros_like(state.psi)
    eta_field = np.full_like(state.psi, state.eta)
    return ControlVector(E_drive=z.copy(), eta_eff=eta_field, F_omega=z.copy())


# ---------------------------------------------------------------------------
# Objective evaluation
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveWeights:
    r"""Coefficients :math:`(\alpha, \beta, \gamma)` of the composite objective
    (problem-statement §3).

    .. math::
        \max_u \int_0^T \big[\alpha\,\mathcal R + \beta\,\mathcal I_{plasmoid}
                              + \gamma\,P\big]\, dt
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


@dataclass
class ObjectiveValue:
    R_term: float
    I_term: float
    P_term: float
    total: float


def reconnection_localization(state: MHDState, target_mask: FloatArray) -> float:
    r""":math:`\mathcal R(x, t)` integrated against a target mask [D].

    ``target_mask`` is a non-negative weight field over the grid (e.g. a Gaussian
    centered on the desired X-point). The localization metric is

    .. math::
        \mathcal R = \int |\eta J_z|\, w_{target}\, dV
    """
    if target_mask.shape != state.psi.shape:
        raise ValueError("target_mask shape must match grid.")
    Jz = state.Jz()
    return float(np.sum(np.abs(state.eta * Jz) * target_mask)) * state.grid.dx * state.grid.dy


def plasmoid_indicator(state: MHDState, k_band: tuple[float, float] = (4.0, 16.0)) -> float:
    r"""Spectral energy in a plasmoid wavenumber band [D].

    Sums :math:`(k_x^2+k_y^2)\,|\hat\psi|^2` over modes with
    :math:`k_{lo} \le \sqrt{k_x^2+k_y^2} \le k_{hi}`. Higher-``k`` content is a
    proxy for plasmoid generation in tearing-unstable sheets [V/D].
    """
    g = state.grid
    KX, KY = g.kgrid()
    K = np.sqrt(KX * KX + KY * KY)
    psi_hat = np.fft.fft2(state.psi) / (g.nx * g.ny)
    Pmag = (KX * KX + KY * KY) * (psi_hat.real ** 2 + psi_hat.imag ** 2)
    k_lo, k_hi = k_band
    if k_lo < 0 or k_hi <= k_lo:
        raise ValueError("Invalid k_band.")
    mask = (k_lo <= K) & (k_hi >= K)
    return float(Pmag[mask].sum())


def energy_release_rate(state: MHDState) -> float:
    r"""Total Ohmic dissipation :math:`P = \int \eta J_z^2\, dV` [V] (matches
    :math:`\mathbf J\!\cdot\!\mathbf E` for the resistive part of Ohm's law)."""
    return state.ohmic_dissipation()


def evaluate_objective(
    state: MHDState,
    weights: ObjectiveWeights,
    target_mask: FloatArray,
    k_band: tuple[float, float] = (4.0, 16.0),
) -> ObjectiveValue:
    """Evaluate the composite objective on the current state."""
    R = reconnection_localization(state, target_mask)
    I = plasmoid_indicator(state, k_band=k_band)
    P = energy_release_rate(state)
    total = weights.alpha * R + weights.beta * I + weights.gamma * P
    return ObjectiveValue(R_term=R, I_term=I, P_term=P, total=total)


# ---------------------------------------------------------------------------
# Simple gradient-shaped controller
# ---------------------------------------------------------------------------


@dataclass
class ControllerConfig:
    """Hyperparameters for :class:`ReconnectionController`.

    Attributes
    ----------
    E_drive_amplitude : float
        Maximum amplitude of the localized reconnection-forcing field
        ``E_drive``. Saturates the actuator to model power limits.
    eta_boost : float
        Factor by which ``η`` is locally elevated inside the target region
        (η_eff = η * (1 + eta_boost * w_target)).
    F_omega_amplitude : float
        Maximum amplitude of the vorticity forcing ``F_ω``.
    weights : ObjectiveWeights
    k_band : tuple[float, float]
        Plasmoid spectral band passed to :func:`plasmoid_indicator`.
    """

    E_drive_amplitude: float = 1.0e-3
    eta_boost: float = 5.0
    F_omega_amplitude: float = 0.0
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    k_band: tuple[float, float] = (4.0, 16.0)


class ReconnectionController:
    r"""CIIR-structured reconnection shaping controller.

    Strategy (interpretive)
    -----------------------
    Given a target mask :math:`w_{target}(x, y) \in [0, 1]`, the controller
    sets:

    - :math:`E_{drive}(x, y) = A_E\,w_{target}\,\mathrm{sign}(J_z)` —
      pumps the reconnection electric field on the target X-point in the
      direction that *amplifies* the existing current there. Sign-following
      avoids inadvertent flux pile-up reversal [D].
    - :math:`\eta_{eff}(x, y) = \eta\,(1 + \kappa\,w_{target})` —
      localizes diffusion to the target region (anomalous-resistivity
      surrogate) [V/D]. Outside the target, base ``η`` is preserved.
    - :math:`F_\omega(x, y) = A_\omega\,w_{target}` if a vortical drive is
      requested.

    This is a heuristic, not optimal control; it is intended as a baseline
    against which RL or adjoint-based optimal-control schemes can be compared
    (see §8.C of the spec).
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()

    def synthesize(self, state: MHDState, target_mask: FloatArray) -> ControlVector:
        """Compute :class:`ControlVector` for the current state and target."""
        if target_mask.shape != state.psi.shape:
            raise ValueError("target_mask shape must match grid.")
        if np.any(target_mask < 0.0):
            raise ValueError("target_mask must be non-negative.")
        # Normalize target mask to [0, 1] for interpretability of amplitudes.
        m = float(target_mask.max())
        w = target_mask / m if m > 0.0 else target_mask

        Jz = state.Jz()
        sign = np.sign(Jz)

        E_drive = self.config.E_drive_amplitude * w * sign
        eta_eff = state.eta * (1.0 + self.config.eta_boost * w)
        F_omega = self.config.F_omega_amplitude * w

        return ControlVector(E_drive=E_drive, eta_eff=eta_eff, F_omega=F_omega)


def gaussian_target_mask(
    state: MHDState, x0: float, y0: float, sigma: float
) -> FloatArray:
    r"""Periodic-aware Gaussian weight :math:`\exp(-r^2/(2\sigma^2))` centered at ``(x0, y0)``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    g = state.grid
    X, Y = g.coords()
    dxv = X - x0
    dyv = Y - y0
    # wrap to [-L/2, L/2]
    dxv = dxv - g.Lx * np.round(dxv / g.Lx)
    dyv = dyv - g.Ly * np.round(dyv / g.Ly)
    r2 = dxv * dxv + dyv * dyv
    return np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float64)


__all__ = [
    "ControlVector",
    "ObjectiveWeights",
    "ObjectiveValue",
    "ControllerConfig",
    "ReconnectionController",
    "zero_control_vector",
    "reconnection_localization",
    "plasmoid_indicator",
    "energy_release_rate",
    "evaluate_objective",
    "gaussian_target_mask",
]

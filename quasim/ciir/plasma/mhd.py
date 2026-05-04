r"""2D reduced resistive MHD scaffold for magnetic reconnection studies.

Physics labels
--------------
[V] = established physics (textbook).
[D] = derived from multiple established results.
[H] = hypothesis (explicitly falsifiable).
CIIR is a structuring lens, not physics.

Model
-----
We integrate the 2D incompressible reduced-MHD (RMHD) equations on a periodic
square domain :math:`[0, L_x) \times [0, L_y)` in the :math:`(x, y)` plane, with
a uniform guide field :math:`B_z \hat z`. The state is

.. math::
    \mathbf{B}_\perp = \nabla \psi \times \hat z,
    \qquad \mathbf{v}_\perp = \hat z \times \nabla \varphi,

so that :math:`\nabla\!\cdot\!\mathbf B_\perp = 0` and
:math:`\nabla\!\cdot\!\mathbf v_\perp = 0` are exact by construction. The
out-of-plane current density is

.. math::
    J_z = -\nabla^2 \psi \quad [V]

and the out-of-plane vorticity is :math:`\omega = \nabla^2 \varphi` [V].

Evolution equations (Strauss 1976 RMHD; see Biskamp, *Magnetic Reconnection
in Plasmas*, 2000) [V]:

.. math::
    \partial_t \psi + \{\varphi, \psi\} = \eta\, \nabla^2 \psi + E_{drive},
    \qquad
    \partial_t \omega + \{\varphi, \omega\} = \{\psi, \nabla^2 \psi\}
        + \nu\, \nabla^2 \omega,

where :math:`\{f, g\} = \partial_x f\,\partial_y g - \partial_y f\,\partial_x g`
is the Poisson bracket [V]. The first equation is the induction equation in
flux form; the second is the vorticity equation (curl of the momentum equation
with the :math:`\mathbf J\!\times\!\mathbf B` term written as
:math:`\{\psi, J_z\}`) [V].

Numerics
--------
- Periodic boundaries; central second-order finite differences on a uniform grid.
- Time integration: explicit RK2 (midpoint) for clarity. CFL is the user's
  responsibility; sensible default ``dt`` is set conservatively in
  :func:`recommended_dt`.
- The :math:`E_{drive}(x, y, t)` term is the **control input** for reconnection
  forcing; :math:`\eta_{eff}(x, y)` is the **control input** for diffusion
  localization. These are passed in by the controller.

Limitations [stated, not hidden]
--------------------------------
- 2D incompressible RMHD: cannot represent fast magnetosonic waves, compressible
  effects, or fully 3D line-tied geometry. It is the standard educational and
  research model for tearing / plasmoid studies in a strongly magnetized
  plasma [V]. For ITER-class predictions one uses M3D-C1, NIMROD, JOREK, etc.
- Hall and electron-inertia terms are *not* included here (pure resistive MHD).
  Adding them would require generalized Ohm's law; see the parent module's
  ``ciir.py`` for the regime classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid2D:
    """Uniform 2D periodic grid on ``[0, Lx) x [0, Ly)``."""

    nx: int
    ny: int
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        if self.nx < 4 or self.ny < 4:
            raise ValueError("Grid must have at least 4 points per dimension.")
        if self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("Domain lengths must be positive.")

    @property
    def dx(self) -> float:
        return float(self.Lx) / self.nx

    @property
    def dy(self) -> float:
        return float(self.Ly) / self.ny

    def coords(self) -> tuple[FloatArray, FloatArray]:
        """Return meshgrid X, Y of node coordinates (indexing='ij')."""
        x = np.linspace(0.0, self.Lx, self.nx, endpoint=False)
        y = np.linspace(0.0, self.Ly, self.ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def kgrid(self) -> tuple[FloatArray, FloatArray]:
        """Return (kx, ky) wavenumber grids consistent with FFT layout."""
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return KX, KY


# ---------------------------------------------------------------------------
# Differential operators (periodic, 2nd-order central FD)
# ---------------------------------------------------------------------------


def ddx(f: FloatArray, dx: float) -> FloatArray:
    """∂_x with periodic boundaries [V]."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)


def ddy(f: FloatArray, dy: float) -> FloatArray:
    """∂_y with periodic boundaries [V]."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)


def laplacian(f: FloatArray, dx: float, dy: float) -> FloatArray:
    r"""5-point Laplacian :math:`\nabla^2 f` with periodic boundaries [V]."""
    fxx = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dx * dx)
    fyy = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dy * dy)
    return fxx + fyy


def poisson_bracket(f: FloatArray, g: FloatArray, dx: float, dy: float) -> FloatArray:
    r"""Arakawa-style Poisson bracket :math:`\{f, g\} = \partial_x f\,\partial_y g
    - \partial_y f\,\partial_x g`.

    We use the simple central-difference form here (sufficient for the scaffold);
    the full Arakawa-J1+J2+J3 conservative form (Arakawa 1966) [V] could replace
    this for energy/enstrophy conservation in long runs.
    """
    return ddx(f, dx) * ddy(g, dy) - ddy(f, dy) * ddx(g, dx)


def invert_laplacian_periodic(rhs: FloatArray, grid: Grid2D) -> FloatArray:
    r"""Solve :math:`\nabla^2 \varphi = \rho` on a periodic grid via FFT [V].

    The k = 0 mode is set to zero (consistent with periodic boundaries; the
    operator has a constant null space). This is used to recover the stream
    function :math:`\varphi` from the vorticity :math:`\omega = \nabla^2 \varphi`.
    """
    KX, KY = grid.kgrid()
    K2 = KX * KX + KY * KY
    rhs_hat = np.fft.fft2(rhs)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_hat = np.where(K2 > 0.0, -rhs_hat / K2, 0.0)
    phi = np.real(np.fft.ifft2(phi_hat))
    return phi


# ---------------------------------------------------------------------------
# State + diagnostics
# ---------------------------------------------------------------------------


@dataclass
class MHDState:
    r"""2D RMHD state.

    Attributes
    ----------
    psi : FloatArray
        Magnetic flux function :math:`\psi(x, y)`. ``B_x = ∂_y ψ``, ``B_y = -∂_x ψ``.
    omega : FloatArray
        Out-of-plane vorticity :math:`\omega = \nabla^2 \varphi`.
    grid : Grid2D
    eta : float
        Background resistivity (Lundquist :math:`S = L V_A / \eta`) [V].
    nu : float
        Kinematic viscosity (Reynolds :math:`Re = L V / \nu`) [V].
    Bz : float
        Uniform guide field magnitude (informational; does not enter 2D RMHD).
    t : float
        Simulation time.
    """

    psi: FloatArray
    omega: FloatArray
    grid: Grid2D
    eta: float = 1.0e-3
    nu: float = 1.0e-3
    Bz: float = 1.0
    t: float = 0.0

    def __post_init__(self) -> None:
        shape = (self.grid.nx, self.grid.ny)
        if self.psi.shape != shape or self.omega.shape != shape:
            raise ValueError(
                f"State shape mismatch: expected {shape}, "
                f"got psi={self.psi.shape}, omega={self.omega.shape}."
            )
        if self.eta < 0.0 or self.nu < 0.0:
            raise ValueError("Diffusivities must be non-negative.")

    # ---- diagnostics -------------------------------------------------------

    def Jz(self) -> FloatArray:
        """Out-of-plane current density :math:`J_z = -\\nabla^2 \\psi` [V]."""
        return -laplacian(self.psi, self.grid.dx, self.grid.dy)

    def phi(self) -> FloatArray:
        r"""Stream function :math:`\varphi` such that :math:`\omega = \nabla^2 \varphi` [V]."""
        return invert_laplacian_periodic(self.omega, self.grid)

    def B_perp(self) -> tuple[FloatArray, FloatArray]:
        """In-plane magnetic field ``(B_x, B_y)`` from ``ψ`` [V]."""
        Bx = ddy(self.psi, self.grid.dy)
        By = -ddx(self.psi, self.grid.dx)
        return Bx, By

    def v_perp(self) -> tuple[FloatArray, FloatArray]:
        """In-plane velocity ``(v_x, v_y)`` from ``φ`` [V]."""
        phi = self.phi()
        vx = -ddy(phi, self.grid.dy)
        vy = ddx(phi, self.grid.dx)
        return vx, vy

    def magnetic_energy(self) -> float:
        r""":math:`E_M = \tfrac{1}{2} \int |\mathbf B_\perp|^2\, dV` [V]."""
        Bx, By = self.B_perp()
        return 0.5 * float(np.sum(Bx * Bx + By * By)) * self.grid.dx * self.grid.dy

    def kinetic_energy(self) -> float:
        r""":math:`E_K = \tfrac{1}{2} \int |\mathbf v_\perp|^2\, dV` [V]."""
        vx, vy = self.v_perp()
        return 0.5 * float(np.sum(vx * vx + vy * vy)) * self.grid.dx * self.grid.dy

    def ohmic_dissipation(self) -> float:
        r""":math:`P_\Omega = \int \eta J_z^2\, dV` [V]."""
        Jz = self.Jz()
        return float(np.sum(self.eta * Jz * Jz)) * self.grid.dx * self.grid.dy

    def divB_max(self) -> float:
        r""":math:`\max |\nabla\!\cdot\!\mathbf B_\perp|` — should be machine-zero
        because :math:`\mathbf B = \nabla \psi \times \hat z` exactly [V]."""
        Bx, By = self.B_perp()
        divB = ddx(Bx, self.grid.dx) + ddy(By, self.grid.dy)
        return float(np.max(np.abs(divB)))

    def copy(self) -> MHDState:
        return MHDState(
            psi=self.psi.copy(),
            omega=self.omega.copy(),
            grid=self.grid,
            eta=self.eta,
            nu=self.nu,
            Bz=self.Bz,
            t=self.t,
        )


# ---------------------------------------------------------------------------
# Equilibria
# ---------------------------------------------------------------------------


def harris_sheet(grid: Grid2D, B0: float = 1.0, a: float = 0.5) -> FloatArray:
    r"""Double Harris-sheet flux function, periodicised [V].

    .. math::
        \psi_0(y) = B_0 a \,\big[\, \ln\cosh((y - L_y/4)/a)
                                 - \ln\cosh((y - 3L_y/4)/a) \,\big] - c_1 y - c_0

    yields a pair of antiparallel current sheets at ``y = L_y/4`` and
    ``y = 3 L_y/4``. The linear and constant terms ``c_1 y + c_0`` are chosen
    to enforce :math:`\psi_0(0) = \psi_0(L_y)` so the function is genuinely
    periodic on the discrete grid. This adds a small uniform :math:`B_x = c_1`
    background; for ``L_y / (4 a) \gg 1`` this background is :math:`O(1)` but
    spatially uniform and does not perturb the X-point / O-point structure.
    Standard test problem for tearing / plasmoid studies (see Loureiro,
    Schekochihin, Cowley 2007 [V]).
    """
    if a <= 0.0:
        raise ValueError("Sheet width 'a' must be positive.")
    _, Y = grid.coords()
    y1 = grid.Ly / 4.0
    y2 = 3.0 * grid.Ly / 4.0
    psi_raw = B0 * a * (np.log(np.cosh((Y - y1) / a)) - np.log(np.cosh((Y - y2) / a)))
    # Remove linear ramp so ψ(0) = ψ(Ly) on the discrete grid (periodicity).
    # The discrete first and last rows in y must match.
    y = np.linspace(0.0, grid.Ly, grid.ny, endpoint=False)
    psi_y = B0 * a * (np.log(np.cosh((y - y1) / a)) - np.log(np.cosh((y - y2) / a)))
    # Linear correction: choose c1 so that the periodic extension psi_y(Ly) ≈ psi_y(0).
    # We approximate psi_y(Ly) by extrapolating the last row using the spacing dy.
    psi_at_Ly = B0 * a * (np.log(np.cosh((grid.Ly - y1) / a)) - np.log(np.cosh((grid.Ly - y2) / a)))
    c1 = (psi_at_Ly - psi_y[0]) / grid.Ly
    psi = psi_raw - c1 * Y
    return psi.astype(np.float64)


def perturb_tearing(
    grid: Grid2D,
    amplitude: float = 1.0e-3,
    mode: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    r"""Tearing-seed flux perturbation :math:`\delta \psi = A \cos(k_x x)
    \,e^{-(y-L_y/4)^2/a_p^2}` localised on the lower sheet [V/D].

    ``mode`` is the integer wavenumber :math:`k_x = 2\pi\,\text{mode}/L_x`.
    A small amount of broadband noise is added if ``rng`` is provided.
    """
    X, Y = grid.coords()
    kx = 2.0 * np.pi * mode / grid.Lx
    a_p = max(grid.Ly / 16.0, 4.0 * grid.dy)
    y1 = grid.Ly / 4.0
    delta = amplitude * np.cos(kx * X) * np.exp(-((Y - y1) ** 2) / (a_p * a_p))
    if rng is not None:
        delta = delta + (amplitude * 1.0e-2) * rng.standard_normal(delta.shape)
    return delta.astype(np.float64)


# ---------------------------------------------------------------------------
# RHS + time stepper
# ---------------------------------------------------------------------------


# Type for the control hook: returns (E_drive, eta_eff, force_vorticity) given (state, t).
ControlHook = Callable[[MHDState, float], tuple[FloatArray, FloatArray, FloatArray]]


def zero_control(state: MHDState, t: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Default no-op control: returns zero forcing fields with the right shape."""
    z = np.zeros_like(state.psi)
    eta_field = np.full_like(state.psi, state.eta)
    return z, eta_field, z


def rhs(
    state: MHDState,
    control: ControlHook = zero_control,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute :math:`(\partial_t \psi, \partial_t \omega)` for the 2D RMHD system [V].

    .. math::
        \partial_t \psi   &= -\{\varphi, \psi\} + \nabla\!\cdot(\eta_{eff}\nabla \psi) + E_{drive} \\
        \partial_t \omega &= -\{\varphi, \omega\} + \{\psi, J_z\} + \nu\,\nabla^2 \omega + F_\omega

    where :math:`F_\omega` represents pressure-driven / external vortical forcing
    (the controller's ``p_drive`` channel, expressed in vorticity form).
    """
    g = state.grid
    dx, dy = g.dx, g.dy

    E_drive, eta_field, F_omega = control(state, state.t)

    phi = state.phi()
    Jz = state.Jz()  # = -∇²ψ

    # Induction: ∂_t ψ = -{φ,ψ} + ∇·(η ∇ψ) + E_drive
    psi_x = ddx(state.psi, dx)
    psi_y = ddy(state.psi, dy)
    flux_x = eta_field * psi_x
    flux_y = eta_field * psi_y
    div_eta_grad_psi = ddx(flux_x, dx) + ddy(flux_y, dy)
    dpsi_dt = -poisson_bracket(phi, state.psi, dx, dy) + div_eta_grad_psi + E_drive

    # Vorticity: ∂_t ω = -{φ,ω} + {ψ,J_z} + ν ∇²ω + F_ω
    domega_dt = (
        -poisson_bracket(phi, state.omega, dx, dy)
        + poisson_bracket(state.psi, Jz, dx, dy)
        + state.nu * laplacian(state.omega, dx, dy)
        + F_omega
    )
    return dpsi_dt, domega_dt


def step_rk2(state: MHDState, dt: float, control: ControlHook = zero_control) -> MHDState:
    """Explicit midpoint (RK2) step. Mutates and returns the same MHDState."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    k1_psi, k1_omega = rhs(state, control)

    mid = state.copy()
    mid.psi = state.psi + 0.5 * dt * k1_psi
    mid.omega = state.omega + 0.5 * dt * k1_omega
    mid.t = state.t + 0.5 * dt
    k2_psi, k2_omega = rhs(mid, control)

    state.psi = state.psi + dt * k2_psi
    state.omega = state.omega + dt * k2_omega
    state.t = state.t + dt
    return state


def recommended_dt(
    grid: Grid2D, eta: float, nu: float, B_max: float = 1.0, safety: float = 0.2
) -> float:
    r"""Conservative time step from CFL + diffusive limits [V/D].

    .. math::
        \Delta t \le \mathrm{safety} \times \min\!\left(
            \frac{\Delta}{V_A^{max}}, \;
            \frac{\Delta^2}{4(\eta + \nu)}
        \right)
    """
    delta = min(grid.dx, grid.dy)
    cfl = delta / max(B_max, 1.0e-12)
    diff_lim = (delta * delta) / (4.0 * (eta + nu + 1.0e-30))
    return float(safety * min(cfl, diff_lim))


__all__ = [
    "Grid2D",
    "MHDState",
    "ControlHook",
    "ddx",
    "ddy",
    "laplacian",
    "poisson_bracket",
    "invert_laplacian_periodic",
    "harris_sheet",
    "perturb_tearing",
    "zero_control",
    "rhs",
    "step_rk2",
    "recommended_dt",
]

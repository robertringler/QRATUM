r"""Instability spectrum tracker 𝕀(k, t).

[V] = established physics. [D] = derived. [H] = hypothesis.

Uses the discrete Fourier transform of :math:`\psi`, :math:`J_z`, and the
in-plane velocity to extract:

- 2D modal energy :math:`E(k_x, k_y, t)` for magnetic and kinetic components.
- Isotropic shell spectrum :math:`E(k, t)` by binning in :math:`k = \sqrt{k_x^2+k_y^2}`.
- Tearing/plasmoid mode bands.

Tearing mode growth rate (Furth–Killeen–Rosenbluth 1963) [V]:

.. math::
    \gamma_{\mathrm{FKR}} \tau_A \sim S^{-3/5} (\Delta')^{4/5}.

Plasmoid instability scaling (Loureiro, Schekochihin, Cowley 2007;
Bhattacharjee et al. 2009) [V]:

.. math::
    \gamma_{\max} \tau_A \sim S^{1/4}, \qquad k_{\max} L \sim S^{3/8},
    \qquad S > S_c \approx 10^4.

These analytic scalings are *not* assumed by the spectrum tracker; they are
provided as helpers (``fkr_growth_rate_estimate``, ``plasmoid_growth_rate_estimate``)
for *post-hoc* comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .mhd import MHDState

FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


@dataclass
class SpectrumSnapshot:
    """One snapshot of the instability spectrum 𝕀(k, t)."""

    t: float
    k_bins: FloatArray  # shell-bin centers
    E_mag: FloatArray  # magnetic energy per bin
    E_kin: FloatArray  # kinetic energy per bin
    Jz_spec: FloatArray  # |Ĵ_z|^2 per bin (current spectrum)
    peak_k_mag: float  # k of max E_mag (excluding k=0)
    peak_k_Jz: float


def _shell_bin(KX: FloatArray, KY: FloatArray, n_bins: int) -> tuple[FloatArray, IntArrayLike]:
    K = np.sqrt(KX * KX + KY * KY)
    k_max = float(K.max())
    edges = np.linspace(0.0, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(K, edges) - 1, 0, n_bins - 1)
    return centers, bin_idx


# numpy typing alias for in-module use
IntArrayLike = NDArray[np.integer]


def spectrum(state: MHDState, n_bins: Optional[int] = None) -> SpectrumSnapshot:
    r"""Compute magnetic, kinetic, and current shell spectra of the current state [V].

    The 2D Parseval-normalised power spectra are computed for ``ψ``, ``φ``, and
    ``J_z``, then summed to recover the energy decomposition

    .. math::
        E_M = \tfrac{1}{2}\sum_k |\hat{\mathbf B}(k)|^2,
        \qquad
        E_K = \tfrac{1}{2}\sum_k |\hat{\mathbf v}(k)|^2,

    using :math:`\hat B_x = i k_y \hat\psi`, :math:`\hat B_y = -i k_x \hat\psi`,
    so that :math:`|\hat{\mathbf B}|^2 = (k_x^2 + k_y^2)\,|\hat\psi|^2` [V].
    Similarly :math:`|\hat{\mathbf v}|^2 = (k_x^2 + k_y^2)\,|\hat\varphi|^2`.
    """
    g = state.grid
    if n_bins is None:
        n_bins = max(8, min(g.nx, g.ny) // 2)
    KX, KY = g.kgrid()
    K2 = KX * KX + KY * KY

    norm = 1.0 / (g.nx * g.ny)
    psi_hat = np.fft.fft2(state.psi) * norm
    phi_hat = np.fft.fft2(state.phi()) * norm
    Jz_hat = np.fft.fft2(state.Jz()) * norm

    # Zero out DC mode (k = 0) — irrelevant to instability spectrum [V].
    psi_hat[0, 0] = 0.0
    phi_hat[0, 0] = 0.0
    Jz_hat[0, 0] = 0.0

    Pmag = K2 * (psi_hat.real ** 2 + psi_hat.imag ** 2)
    Pkin = K2 * (phi_hat.real ** 2 + phi_hat.imag ** 2)
    PJz = Jz_hat.real ** 2 + Jz_hat.imag ** 2

    centers, bin_idx = _shell_bin(KX, KY, n_bins)
    E_mag = np.zeros(n_bins)
    E_kin = np.zeros(n_bins)
    Jz_spec = np.zeros(n_bins)
    flat_idx = bin_idx.ravel()
    np.add.at(E_mag, flat_idx, Pmag.ravel())
    np.add.at(E_kin, flat_idx, Pkin.ravel())
    np.add.at(Jz_spec, flat_idx, PJz.ravel())

    # Identify dominant mode (DC already removed; argmax across all bins).
    if n_bins >= 1:
        peak_mag = int(np.argmax(E_mag))
        peak_J = int(np.argmax(Jz_spec))
    else:
        peak_mag = 0
        peak_J = 0

    return SpectrumSnapshot(
        t=state.t,
        k_bins=centers,
        E_mag=E_mag,
        E_kin=E_kin,
        Jz_spec=Jz_spec,
        peak_k_mag=float(centers[peak_mag]),
        peak_k_Jz=float(centers[peak_J]),
    )


def growth_rates(snapshots: list[SpectrumSnapshot], eps: float = 1.0e-30) -> FloatArray:
    r"""Empirical per-bin growth rate :math:`\gamma(k)` from a time series of spectra [D].

    Linear regression of :math:`\log E_{mag}(k, t)` vs ``t`` for each bin. The
    return shape is ``(n_bins,)``. Bins whose energy drops below ``eps`` are
    set to ``NaN``.
    """
    if len(snapshots) < 2:
        raise ValueError("Need at least two spectrum snapshots to estimate growth rates.")
    ts = np.array([s.t for s in snapshots], dtype=np.float64)
    n_bins = snapshots[0].k_bins.size
    gammas = np.full(n_bins, np.nan, dtype=np.float64)
    for kb in range(n_bins):
        E = np.array([s.E_mag[kb] for s in snapshots], dtype=np.float64)
        if not np.all(E > eps):
            continue
        logE = np.log(E)
        # Slope of log(E) vs t = 2 γ (since E ∝ A^2).
        slope, _ = np.polyfit(ts, logE, 1)
        gammas[kb] = 0.5 * slope
    return gammas


# ---------------------------------------------------------------------------
# Analytic-scaling reference helpers (post-hoc only — not used in evolution)
# ---------------------------------------------------------------------------


def lundquist_number(state: MHDState, B_char: float = 1.0, L: Optional[float] = None) -> float:
    r"""Lundquist number :math:`S = L V_A / \eta` with :math:`V_A = B_{char}` (ρ ≡ 1) [V]."""
    if L is None:
        L = min(state.grid.Lx, state.grid.Ly) / 4.0  # default: sheet length scale
    if state.eta <= 0.0:
        return float("inf")
    return float(L * B_char / state.eta)


def fkr_growth_rate_estimate(S: float, delta_prime: float = 1.0, tau_A: float = 1.0) -> float:
    r"""FKR scaling :math:`\gamma_{FKR} \sim \tau_A^{-1} S^{-3/5} (\Delta')^{4/5}` [V].

    This is the *standard* analytic scaling, returned as a reference value; the
    actual numerical growth rate is obtained from :func:`growth_rates`.
    """
    if S <= 0.0:
        raise ValueError("S must be positive.")
    if tau_A <= 0.0:
        raise ValueError("tau_A must be positive.")
    return (1.0 / tau_A) * S ** (-3.0 / 5.0) * (max(delta_prime, 0.0)) ** (4.0 / 5.0)


def plasmoid_growth_rate_estimate(S: float, tau_A: float = 1.0, S_c: float = 1.0e4) -> Optional[float]:
    r"""Plasmoid scaling :math:`\gamma_{\max} \sim \tau_A^{-1} S^{1/4}` for :math:`S > S_c` [V].

    Returns ``None`` if ``S < S_c`` (plasmoid regime not reached).
    """
    if S <= 0.0:
        raise ValueError("S must be positive.")
    if S < S_c:
        return None
    return (1.0 / tau_A) * S ** 0.25


__all__ = [
    "SpectrumSnapshot",
    "spectrum",
    "growth_rates",
    "lundquist_number",
    "fkr_growth_rate_estimate",
    "plasmoid_growth_rate_estimate",
]

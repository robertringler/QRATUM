r"""Stability & Analysis Agent.

Lyapunov function
-----------------
V(x, ρ) = ‖x - x*‖² + S(ρ ‖ ρ*)

where S(ρ ‖ ρ*) = Tr(ρ log ρ - ρ log ρ*) is the quantum relative entropy.

Convergence
-----------
We compute dV/dt numerically along trajectories and check dV/dt ≤ 0.

Spectral gap
------------
The Lindbladian superoperator L acts on vec(ρ) as a (d² × d²) matrix.
The spectral gap Δ = min_{Re(λ) < 0} |Re(λ)| determines the relaxation rate.
A larger gap implies faster convergence to the steady state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]
RMatrix = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Lindbladian superoperator matrix
# ---------------------------------------------------------------------------

def lindbladian_matrix(
    H: CMatrix, lindblad_ops: List[CMatrix], hbar: float = 1.0
) -> CMatrix:
    r"""Build the (d², d²) Lindbladian superoperator L in vec(ρ) representation.

    dρ/dt = L[ρ]  ←→  d vec(ρ)/dt = M · vec(ρ)

    Using the identity:
        vec(A ρ B) = (B^T ⊗ A) · vec(ρ)

    L[ρ] = -i/ℏ [H, ρ] + Σ_k (L_k ⊗ L_k* - ½ L_k†L_k ⊗ I - ½ I ⊗ L_k†L_k^T)
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)

    # Coherent part: -i/ℏ (H ⊗ I - I ⊗ H^T)
    M = (-1j / hbar) * (np.kron(H, I) - np.kron(I, H.T))

    for L in lindblad_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        # L ρ L† → (L* ⊗ L) vec(ρ)  ... but with correct convention
        # Standard: vec(L ρ L†) = (L^* ⊗ L) vec(ρ)
        M += np.kron(L.conj(), L)
        # -½ L†L ρ  → -(½) (I ⊗ L†L) vec(ρ)
        M -= 0.5 * np.kron(I, LdL)
        # -½ ρ L†L  → -(½) (L†L^T ⊗ I) vec(ρ)
        M -= 0.5 * np.kron(LdL.T, I)

    return M


def spectral_gap(
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    hbar: float = 1.0,
) -> Tuple[float, NDArray]:
    """Compute the spectral gap of the Lindbladian superoperator.

    Parameters
    ----------
    H           : Hamiltonian
    lindblad_ops: Lindblad operators
    hbar        : ℏ

    Returns
    -------
    gap : float — minimum |Re(λ)| among eigenvalues with Re(λ) < 0
    eigenvalues : (d²,) complex array
    """
    M = lindbladian_matrix(H, lindblad_ops, hbar)
    eigvals = np.linalg.eigvals(M)
    neg_real = eigvals[eigvals.real < -1e-10]
    if len(neg_real) == 0:
        return (0.0, eigvals)
    gap = float(np.min(np.abs(neg_real.real)))
    return (gap, eigvals)


# ---------------------------------------------------------------------------
# Lyapunov analyzer
# ---------------------------------------------------------------------------

@dataclass
class LyapunovResult:
    """Result of a Lyapunov stability analysis."""

    V_traj: NDArray            # V(t)
    dV_traj: NDArray           # numerical dV/dt
    converged: bool
    convergence_time: float    # first time V < V_threshold
    x_norm2_traj: NDArray
    S_rel_traj: NDArray
    spectral_gap: float


class LyapunovAnalyzer:
    r"""Lyapunov stability analysis for the hybrid CIIR controller.

    V(x, ρ) = ‖x - x*‖² + S(ρ ‖ ρ*)

    Convergence criterion: V(t) → 0 as t → ∞.

    Parameters
    ----------
    x_star    : classical equilibrium
    rho_star  : quantum target state
    threshold : V < threshold is deemed "converged"
    """

    def __init__(
        self,
        x_star: Optional[RMatrix] = None,
        rho_star: Optional[CMatrix] = None,
        threshold: float = 0.1,
    ) -> None:
        self.x_star = x_star
        self.rho_star = rho_star
        self.threshold = threshold

    def analyze(
        self,
        times: NDArray,
        x_traj: List[RMatrix],
        rho_traj: List[CMatrix],
        H: Optional[CMatrix] = None,
        lindblad_ops: Optional[List[CMatrix]] = None,
    ) -> LyapunovResult:
        """Compute Lyapunov trajectory and convergence properties.

        Parameters
        ----------
        times       : time array
        x_traj      : classical state trajectory
        rho_traj    : quantum state trajectory
        H           : Hamiltonian (for spectral gap, optional)
        lindblad_ops: Lindblad operators (for spectral gap, optional)
        """
        n = len(times)
        x_star = self.x_star if self.x_star is not None else np.zeros_like(x_traj[0])
        rho_star = self.rho_star if self.rho_star is not None else rho_traj[-1]

        V_traj = np.zeros(n)
        x_norm2_traj = np.zeros(n)
        S_rel_traj = np.zeros(n)

        for i, (x, rho) in enumerate(zip(x_traj, rho_traj)):
            dx = x - x_star
            x_n2 = float(np.dot(dx, dx))
            S = _quantum_relative_entropy(rho, rho_star)
            V_traj[i] = x_n2 + S
            x_norm2_traj[i] = x_n2
            S_rel_traj[i] = S

        # Numerical dV/dt
        dV_traj = np.gradient(V_traj, times)

        # Convergence
        below = np.where(V_traj < self.threshold)[0]
        converged = len(below) > 0
        conv_time = float(times[below[0]]) if converged else float(times[-1])

        # Spectral gap
        gap = 0.0
        if H is not None and lindblad_ops is not None:
            gap, _ = spectral_gap(H, lindblad_ops)

        return LyapunovResult(
            V_traj=V_traj,
            dV_traj=dV_traj,
            converged=converged,
            convergence_time=conv_time,
            x_norm2_traj=x_norm2_traj,
            S_rel_traj=S_rel_traj,
            spectral_gap=gap,
        )


def _quantum_relative_entropy(rho: CMatrix, sigma: CMatrix) -> float:
    """S(ρ‖σ) = Tr(ρ log ρ) - Tr(ρ log σ) ≥ 0."""
    eps = 1e-12
    try:
        evals_r, evecs_r = np.linalg.eigh(rho)
        evals_r = np.clip(evals_r.real, eps, None)
        log_rho = evecs_r @ np.diag(np.log(evals_r)) @ evecs_r.conj().T

        evals_s, evecs_s = np.linalg.eigh(sigma)
        evals_s = np.clip(evals_s.real, eps, None)
        log_sigma = evecs_s @ np.diag(np.log(evals_s)) @ evecs_s.conj().T

        S = float(np.real(np.trace(rho @ log_rho) - np.trace(rho @ log_sigma)))
        return max(0.0, S)
    except Exception:
        return 0.0

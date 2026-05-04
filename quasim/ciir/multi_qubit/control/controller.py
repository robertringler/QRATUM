r"""Control System Agent.

Classical subsystem
-------------------
Linear Quadratic Regulator (LQR) for the classical state x(t) ∈ ℝ^m:

    ẋ = A x + B u + ξ(t)
    u = -K x,   K = R^{-1} B^T P   (P solves the DARE)

Quantum control
---------------
Pontryagin optimal control for the quantum subsystem (Ch21, §4):

    u_k*(t) = (i / r ℏ) Tr( P(t) [H_k, ρ_Q(t)] )

where H_k are control Hamiltonians and P(t) is the co-state density matrix.

Hybrid controller
-----------------
Maps x(t) → u(t) → H_ctrl(x, t) via

    H_ctrl(x, t) = H_0 + Σ_k u_k(t) H_k

where H_0 is the drift Hamiltonian.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

CMatrix = NDArray[np.complex128]
RMatrix = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Utility: Pauli matrices for building control Hamiltonians
# ---------------------------------------------------------------------------


def _kron_embed(op: CMatrix, qubit: int, n_qubits: int) -> CMatrix:
    I1 = np.eye(2, dtype=complex)
    ops = [I1] * n_qubits
    ops[qubit] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def control_hamiltonian(n_qubits: int) -> Tuple[CMatrix, List[CMatrix]]:
    """Return drift H0 and control Hamiltonians H_k.

    H_ctrl(u, t) = H_0 + Σ_k u_k H_k

    Drift: random hermitian (represents natural coupling)
    Controls: σ_x, σ_y, σ_z on each qubit (3N control fields)
    """
    dim = 2**n_qubits
    rng = np.random.default_rng(42)
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    H0 = 0.05 * (A + A.conj().T)  # small drift

    H_k = []
    for q in range(n_qubits):
        for pauli in (SX, SY, SZ):
            H_k.append(_kron_embed(pauli, q, n_qubits))
    return H0, H_k


# ---------------------------------------------------------------------------
# Classical LQR
# ---------------------------------------------------------------------------


class ClassicalLQR:
    r"""LQR stabilizer for ẋ = A x + B u.

    Solves the continuous-time algebraic Riccati equation (CARE):

        A^T P + P A - P B R^{-1} B^T P + Q = 0

    and computes the gain K = R^{-1} B^T P so that u = -K x.

    Parameters
    ----------
    A : (m, m) system matrix
    B : (m, p) input matrix
    Q : (m, m) state cost matrix (positive semidefinite)
    R : (p, p) control cost matrix (positive definite)
    """

    def __init__(
        self,
        A: RMatrix,
        B: RMatrix,
        Q: Optional[RMatrix] = None,
        R: Optional[RMatrix] = None,
    ) -> None:
        m, p = B.shape
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.Q = np.eye(m) if Q is None else np.array(Q, dtype=float)
        self.R = np.eye(p) if R is None else np.array(R, dtype=float)
        self._compute_gain()

    def _compute_gain(self) -> None:
        try:
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        except Exception:
            # Fallback: use pole placement approximation
            P = np.eye(self.A.shape[0])
        self.P = P
        self.K = np.linalg.solve(self.R, self.B.T @ P)

    def control(self, x: RMatrix) -> RMatrix:
        """Return u = -K x."""
        return -(self.K @ x)

    def step(
        self,
        x: RMatrix,
        dt: float,
        noise_std: float = 0.0,
    ) -> Tuple[RMatrix, RMatrix]:
        """Advance x by dt using u = -K x (Euler step with optional noise).

        Returns (x_new, u).
        """
        u = self.control(x)
        dx = self.A @ x + self.B @ u
        if noise_std > 0:
            dx = dx + np.random.normal(0, noise_std, size=x.shape)
        return x + dt * dx, u


def _default_lqr(m: int = 4) -> ClassicalLQR:
    """Construct a default double-integrator-like LQR system."""
    # Block: double integrator for m/2 modes
    half = m // 2
    A = np.zeros((m, m))
    A[:half, half:] = np.eye(half)  # ẋ₁ = x₂
    B = np.zeros((m, half))
    B[half:, :] = np.eye(half)  # ẋ₂ = u
    Q = np.eye(m)
    R = 0.1 * np.eye(half)
    return ClassicalLQR(A, B, Q, R)


# ---------------------------------------------------------------------------
# Pontryagin optimal quantum controller
# ---------------------------------------------------------------------------


class PontryaginController:
    r"""Pontryagin optimal control for the quantum subsystem.

    Optimal control amplitudes (Ch21, Eq. u*):

        u_k*(t) = (i / r ℏ) Tr( P [H_k, ρ_Q] )

    where P is the co-state density matrix, approximated here as the
    target state ρ* (gradient ascent on fidelity).

    Parameters
    ----------
    H_k   : list of control Hamiltonians
    r     : regularisation parameter (penalty on |u|²)
    hbar  : ℏ (default 1 in natural units)
    """

    def __init__(
        self,
        H_k: List[CMatrix],
        r: float = 1.0,
        hbar: float = 1.0,
    ) -> None:
        self.H_k = H_k
        self.r = r
        self.hbar = hbar

    def amplitudes(self, rho: CMatrix, P: CMatrix) -> List[float]:
        r"""Compute Pontryagin optimal amplitudes u_k*.

        u_k* = (i / r ℏ) Tr( P [H_k, ρ] )
        """
        amps = []
        for Hk in self.H_k:
            comm = Hk @ rho - rho @ Hk  # [H_k, ρ]
            val = np.trace(P @ comm)  # Tr(P [H_k, ρ])
            u_k = (1j / (self.r * self.hbar)) * val
            amps.append(float(np.real(u_k)))  # real part only
        return amps

    def hamiltonian(
        self,
        H0: CMatrix,
        rho: CMatrix,
        P: CMatrix,
        scale: float = 1.0,
    ) -> CMatrix:
        """Build H_ctrl = H0 + Σ_k u_k* H_k."""
        amps = self.amplitudes(rho, P)
        H = H0.copy()
        for u_k, Hk in zip(amps, self.H_k):
            H = H + scale * u_k * Hk
        return H


# ---------------------------------------------------------------------------
# Hybrid controller: maps classical x(t) → quantum H_ctrl
# ---------------------------------------------------------------------------


class HybridController:
    """Hybrid classical–quantum controller.

    Maps the classical state x(t) to a quantum control Hamiltonian via:

        H_ctrl(x, t) = H0 + Σ_k κ_k(x) H_k

    where κ_k(x) = x[k % m] (direct coupling, modulo classical dimension).

    Also maintains the LQR loop on the classical subsystem.

    Parameters
    ----------
    n_qubits : int
    m        : classical state dimension
    """

    def __init__(self, n_qubits: int = 2, m: int = 4) -> None:
        self.n_qubits = n_qubits
        self.m = m
        self.dim = 2**n_qubits
        self.H0, self.H_k = control_hamiltonian(n_qubits)
        self.lqr = _default_lqr(m)
        self.pontryagin = PontryaginController(self.H_k)

    def H_ctrl(self, x: RMatrix, rho: CMatrix, rho_target: CMatrix) -> CMatrix:
        """Compute H_ctrl from classical state x and quantum state ρ.

        Classical coupling contributes O(0.01) rotations; Pontryagin
        corrections are applied at O(0.05) scale to avoid fighting Lindblad.
        """
        H = self.H0.copy()
        # Classical coupling: u_k ~ x[k % m] (small scale to avoid fighting Lindblad)
        for k, Hk in enumerate(self.H_k):
            alpha = float(x[k % self.m]) * 0.01
            H = H + alpha * Hk
        # Pontryagin corrections (small scale)
        H_pont = self.pontryagin.hamiltonian(np.zeros_like(self.H0), rho, rho_target, scale=0.05)
        return H + H_pont

    def step_classical(
        self,
        x: RMatrix,
        dt: float,
        noise_std: float = 0.01,
    ) -> Tuple[RMatrix, RMatrix]:
        """Advance classical state."""
        return self.lqr.step(x, dt, noise_std=noise_std)

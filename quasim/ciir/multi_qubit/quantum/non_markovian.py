r"""Non-Markovian Lindblad Evolution — GAP 6 closure.

Replaces the Markovian GKSL equation

    dρ/dt = L₀(ρ)

with the generalized Nakajima–Zwanzig memory-kernel equation:

    dρ/dt = L₀(ρ(t)) + ∫₀ᵗ K(t−s) · L₁(ρ(s)) ds

where

* L₀ is the standard Markovian GKSL superoperator (amplitude damping,
  dephasing, depolarizing);
* K(t−s) is a scalar memory kernel encoding non-Markovian backflow;
* L₁ is the "memory Lindbladian" (same jump operators as L₀, modulated
  by the kernel weight).

Two kernel types are implemented:

    Exponential  : K(t) = λ · exp(−λ t)           (Lorentzian bath)
    Oscillatory  : K(t) = exp(−γ t) · cos(ω t)    (under-damped bath)

Numerical scheme
----------------
History of ρ is stored at every integration step.  The memory integral

    ∫₀ᵗ K(t−s) L₁(ρ(s)) ds

is evaluated by the trapezoidal rule over all stored samples, then added
as a correction to the Markovian RK4 derivative.

Markovian limit
---------------
For K(t) = δ(t) (very large λ) the equation reduces to ordinary GKSL.
The module exposes a helper to verify this limit numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]

# ---------------------------------------------------------------------------
# Memory kernel
# ---------------------------------------------------------------------------


@dataclass
class MemoryKernel:
    """Scalar memory kernel K(t).

    Parameters
    ----------
    kernel_type : ``"exponential"`` or ``"oscillatory"``
    lambda_     : decay rate for exponential kernel  (units: 1/time)
    gamma       : damping rate for oscillatory kernel (units: 1/time)
    omega       : oscillation frequency              (units: 1/time)
    strength    : overall prefactor κ scaling K(t)
    """

    kernel_type: str = "exponential"
    lambda_: float = 1.0
    gamma: float = 0.5
    omega: float = 2.0
    strength: float = 0.1

    def __call__(self, t: float) -> float:
        """Evaluate K(t) for t ≥ 0."""
        if t < 0.0:
            return 0.0
        if self.kernel_type == "exponential":
            return self.strength * self.lambda_ * np.exp(-self.lambda_ * t)
        elif self.kernel_type == "oscillatory":
            return self.strength * np.exp(-self.gamma * t) * np.cos(self.omega * t)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type!r}")

    def norm(self, t_max: float = 50.0, n_points: int = 10000) -> float:
        r"""∫₀^{t_max} |K(t)| dt  (approximate L1 norm via trapezoidal rule)."""
        ts = np.linspace(0, t_max, n_points)
        vals = np.array([abs(self(t)) for t in ts])
        # Manual trapezoidal rule for numpy 2.x compatibility
        dt = ts[1] - ts[0]
        return float(dt * (0.5 * vals[0] + np.sum(vals[1:-1]) + 0.5 * vals[-1]))


# ---------------------------------------------------------------------------
# Non-Markovian dissipator
# ---------------------------------------------------------------------------

def _lindblad_dissipator(rho: CMatrix, lindblad_ops: List[CMatrix]) -> CMatrix:
    """Standard Lindblad dissipator Σ_k D[L_k]ρ."""
    out = np.zeros_like(rho)
    for L in lindblad_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        out += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
    return out


def _gksl_rhs_nm(
    rho: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    memory_term: CMatrix,
    hbar: float = 1.0,
) -> CMatrix:
    """GKSL RHS including pre-computed memory correction."""
    coherent = -1j / hbar * (H @ rho - rho @ H)
    markov_diss = _lindblad_dissipator(rho, lindblad_ops)
    return coherent + markov_diss + memory_term


# ---------------------------------------------------------------------------
# Non-Markovian evolver
# ---------------------------------------------------------------------------

class NonMarkovianEvolver:
    r"""Time-evolve ρ under a memory-kernel non-Markovian GKSL equation.

    dρ/dt = L₀(ρ(t)) + ∫₀ᵗ K(t−s) L₁(ρ(s)) ds

    where L₁ is the same Lindblad dissipator as L₀ but weighted by K.

    Parameters
    ----------
    H           : control Hamiltonian (constant during evolution)
    lindblad_ops: Lindblad jump operators
    kernel      : MemoryKernel instance
    dt          : time step
    hbar        : ℏ (default 1)
    """

    def __init__(
        self,
        H: CMatrix,
        lindblad_ops: List[CMatrix],
        kernel: MemoryKernel,
        dt: float = 0.01,
        hbar: float = 1.0,
    ) -> None:
        self.H = H
        self.lindblad_ops = lindblad_ops
        self.kernel = kernel
        self.dt = dt
        self.hbar = hbar

        # History: list of (time, rho) pairs
        self._history: List[CMatrix] = []
        self._times: List[float] = []

    def _compute_memory_integral(self, t_now: float) -> CMatrix:
        r"""∫₀ᵗ K(t−s) L₁(ρ(s)) ds via trapezoidal rule."""
        if len(self._history) == 0:
            return np.zeros_like(self.H)

        n = len(self._history)
        dt = self.dt
        dissipators = []
        kernel_vals = []

        for i, (t_s, rho_s) in enumerate(zip(self._times, self._history)):
            tau = t_now - t_s
            k_val = self.kernel(tau)
            kernel_vals.append(k_val)
            dissipators.append(_lindblad_dissipator(rho_s, self.lindblad_ops))

        # Trapezoidal weights
        weights = np.ones(n) * dt
        if n >= 2:
            weights[0] *= 0.5
            weights[-1] *= 0.5

        result = np.zeros_like(self.H)
        for w, k, D in zip(weights, kernel_vals, dissipators):
            result += w * k * D

        return result

    def evolve(
        self,
        rho0: CMatrix,
        n_steps: int,
        record_every: int = 1,
    ) -> List[CMatrix]:
        """Integrate non-Markovian GKSL over n_steps time steps.

        Parameters
        ----------
        rho0        : initial density matrix
        n_steps     : number of dt-steps
        record_every: save every this many steps

        Returns
        -------
        list of recorded density matrices (includes rho0)
        """
        rho = rho0.astype(complex).copy()
        self._history = [rho.copy()]
        self._times = [0.0]

        trajectory = [rho.copy()]
        H = self.H
        ops = self.lindblad_ops
        dt = self.dt
        hbar = self.hbar

        for step in range(1, n_steps + 1):
            t_now = step * dt

            # Memory term from history
            mem = self._compute_memory_integral(t_now)

            # RK4 with memory term held fixed over this step
            def f(r: CMatrix) -> CMatrix:
                return _gksl_rhs_nm(r, H, ops, mem, hbar)

            k1 = f(rho)
            k2 = f(rho + 0.5 * dt * k1)
            k3 = f(rho + 0.5 * dt * k2)
            k4 = f(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Enforce Hermiticity, positivity, unit trace
            rho = 0.5 * (rho + rho.conj().T)
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals = np.clip(eigvals.real, 0.0, None)
            rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            tr = float(np.trace(rho).real)
            if tr > 0.0:
                rho /= tr

            # Update history
            self._history.append(rho.copy())
            self._times.append(t_now)

            if step % record_every == 0:
                trajectory.append(rho.copy())

        return trajectory


# ---------------------------------------------------------------------------
# Markovian limit check
# ---------------------------------------------------------------------------

def markovian_limit_rhs(
    rho: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    kernel: MemoryKernel,
    hbar: float = 1.0,
) -> CMatrix:
    r"""GKSL + leading memory correction at t→0.

    For K(t) = κ δ(t) (sharp kernel), the memory integral reduces to
    κ · L₁(ρ(t)).  This function computes the effective GKSL with the
    renormalized rate γ_eff = γ + κ·∫K(s)ds, providing a Markovian
    approximation to the non-Markovian dynamics.
    """
    kernel_norm = kernel.norm()
    coherent = -1j / hbar * (H @ rho - rho @ H)
    diss_markov = _lindblad_dissipator(rho, lindblad_ops)
    diss_memory = _lindblad_dissipator(rho, lindblad_ops)
    return coherent + diss_markov + kernel_norm * diss_memory

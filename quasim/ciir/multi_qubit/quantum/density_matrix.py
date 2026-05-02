r"""Density-matrix Lindblad simulator — Quantum Simulation Agent.

Implements the Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) master equation:

    dρ/dt = -i/ℏ [H_ctrl(x,t), ρ]
            + Σ_k γ_k ( L_k ρ L_k† - ½ {L_k† L_k, ρ} )

for an N-qubit open quantum system.  The density matrix ρ is stored as a
(2^N × 2^N) complex matrix; positivity and unit trace are enforced after
each Runge–Kutta step.

Noise channels supported
------------------------
* Amplitude damping   : L_k = √γ |0⟩⟨1| on each qubit  (T1 relaxation)
* Dephasing           : L_k = √γ σ_z on each qubit       (T2 dephasing)
* Depolarizing        : L_k = √(γ/4){σ_x, σ_y, σ_z, I}  (isotropic)

All operators are returned as sparse-friendly dense NumPy arrays for
compatibility with the rest of the CIIR simulation stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
CMatrix = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Single-qubit Pauli & ladder operators
# ---------------------------------------------------------------------------

_I1 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)  # |0⟩⟨1| = σ_-
_SP = np.array([[0, 0], [1, 0]], dtype=complex)  # |1⟩⟨0| = σ_+


def _kron_embed(op: CMatrix, qubit: int, n_qubits: int) -> CMatrix:
    """Embed a single-qubit operator into the N-qubit tensor product."""
    ops = [_I1] * n_qubits
    ops[qubit] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


# ---------------------------------------------------------------------------
# Lindblad jump operators
# ---------------------------------------------------------------------------

def amplitude_damping_ops(n_qubits: int, gamma: float) -> List[CMatrix]:
    r"""Return amplitude-damping Lindblad operators √γ·σ_- for each qubit.

    Parameters
    ----------
    n_qubits : int
    gamma    : float — relaxation rate (1/T1)

    Returns
    -------
    list of (2^N, 2^N) complex arrays
    """
    sq = np.sqrt(gamma)
    return [sq * _kron_embed(_SM, k, n_qubits) for k in range(n_qubits)]


def dephasing_ops(n_qubits: int, gamma: float) -> List[CMatrix]:
    r"""Return dephasing Lindblad operators √(γ/2)·σ_z for each qubit."""
    sq = np.sqrt(gamma / 2.0)
    return [sq * _kron_embed(_SZ, k, n_qubits) for k in range(n_qubits)]


def depolarizing_ops(n_qubits: int, gamma: float) -> List[CMatrix]:
    r"""Return depolarizing Lindblad operators (σ_x, σ_y, σ_z) per qubit.

    The standard single-qubit depolarizing channel rate p per gate maps to
    γ = p / dt.  The three operators √(γ/4)·{σ_x, σ_y, σ_z} together
    implement the isotropic depolarizing noise at rate γ.
    """
    sq = np.sqrt(gamma / 4.0)
    ops = []
    for k in range(n_qubits):
        for pauli in (_SX, _SY, _SZ):
            ops.append(sq * _kron_embed(pauli, k, n_qubits))
    return ops


# ---------------------------------------------------------------------------
# Lindblad dissipator
# ---------------------------------------------------------------------------

def lindblad_superop(rho: CMatrix, lindblad_ops: List[CMatrix]) -> CMatrix:
    r"""Compute the Lindblad dissipator Σ_k D[L_k]ρ.

    D[L]ρ = L ρ L† - ½ {L†L, ρ}

    Parameters
    ----------
    rho         : (d, d) complex array
    lindblad_ops: list of (d, d) complex arrays

    Returns
    -------
    (d, d) complex array — total dissipative contribution
    """
    out = np.zeros_like(rho)
    for L in lindblad_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        out += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
    return out


# ---------------------------------------------------------------------------
# GKSL right-hand side
# ---------------------------------------------------------------------------

def _gksl_rhs(
    rho: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    hbar: float = 1.0,
) -> CMatrix:
    """Full GKSL/Lindblad right-hand side dρ/dt."""
    coherent = -1j / hbar * (H @ rho - rho @ H)
    dissipative = lindblad_superop(rho, lindblad_ops)
    return coherent + dissipative


# ---------------------------------------------------------------------------
# Runge–Kutta 4 integrator
# ---------------------------------------------------------------------------

def evolve_rk4(
    rho0: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    dt: float,
    n_steps: int,
    hbar: float = 1.0,
    record_every: int = 1,
) -> List[CMatrix]:
    """Integrate the Lindblad equation via fourth-order Runge–Kutta.

    Parameters
    ----------
    rho0        : initial density matrix
    H           : control Hamiltonian (may be updated externally per step)
    lindblad_ops: list of Lindblad operators
    dt          : time step
    n_steps     : number of RK4 steps
    hbar        : reduced Planck constant (default 1 in natural units)
    record_every: save density matrix every this many steps

    Returns
    -------
    list of density matrices at recorded steps (includes rho0)
    """
    rho = rho0.astype(complex)
    dim = rho.shape[0]
    trajectory = [rho.copy()]

    def f(r: CMatrix) -> CMatrix:
        return _gksl_rhs(r, H, lindblad_ops, hbar)

    for step in range(1, n_steps + 1):
        k1 = f(rho)
        k2 = f(rho + 0.5 * dt * k1)
        k3 = f(rho + 0.5 * dt * k2)
        k4 = f(rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Enforce hermiticity & positivity
        rho = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.clip(eigvals.real, 0, None)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        tr = np.trace(rho).real
        if tr > 0:
            rho /= tr
        if step % record_every == 0:
            trajectory.append(rho.copy())

    return trajectory


# ---------------------------------------------------------------------------
# Parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class LindbladParams:
    """Physical noise parameters for the Lindblad solver."""

    n_qubits: int = 2
    gamma_ad: float = 0.01   # amplitude-damping rate (1/T1)
    gamma_dp: float = 0.005  # dephasing rate (1/2T2 - 1/2T1)
    gamma_dep: float = 0.0   # depolarizing rate
    hbar: float = 1.0

    def build_ops(self) -> List[CMatrix]:
        """Construct the full list of Lindblad operators."""
        ops: List[CMatrix] = []
        if self.gamma_ad > 0:
            ops += amplitude_damping_ops(self.n_qubits, self.gamma_ad)
        if self.gamma_dp > 0:
            ops += dephasing_ops(self.n_qubits, self.gamma_dp)
        if self.gamma_dep > 0:
            ops += depolarizing_ops(self.n_qubits, self.gamma_dep)
        return ops


# ---------------------------------------------------------------------------
# High-level simulator class
# ---------------------------------------------------------------------------

class DensityMatrixSimulator:
    """High-level wrapper for multi-qubit Lindblad simulation.

    Supports N = 2 … 6 qubits.

    Parameters
    ----------
    params : LindbladParams
    """

    def __init__(self, params: LindbladParams) -> None:
        if not (2 <= params.n_qubits <= 6):
            raise ValueError("n_qubits must be between 2 and 6.")
        self.params = params
        self.dim = 2 ** params.n_qubits
        self._ops = params.build_ops()

    @property
    def n_qubits(self) -> int:
        return self.params.n_qubits

    def ground_state(self) -> CMatrix:
        """Return |00…0⟩⟨00…0| density matrix."""
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def maximally_mixed(self) -> CMatrix:
        """Return maximally mixed state I/d."""
        return np.eye(self.dim, dtype=complex) / self.dim

    def bell_state(self) -> CMatrix:
        r"""Return |Φ+⟩⟨Φ+| (two-qubit Bell state, first two qubits)."""
        if self.n_qubits < 2:
            raise ValueError("Bell state requires at least 2 qubits.")
        phi_plus = np.zeros(self.dim, dtype=complex)
        phi_plus[0] = 1.0 / np.sqrt(2)   # |00…0⟩
        phi_plus[self.dim // 2 + 1] = 1.0 / np.sqrt(2)  # |11…0⟩ approx
        # Correct: |Φ+⟩ = (|00⟩+|11⟩)/√2 embedded in full space
        phi_plus = np.zeros(self.dim, dtype=complex)
        phi_plus[0] = 1.0 / np.sqrt(2)      # |00⟩ ⊗ |0…0⟩
        phi_plus[3 * (self.dim // 4)] = 1.0 / np.sqrt(2)  # |11⟩ ⊗ |0…0⟩
        # For 2 qubits: indices 0=|00⟩, 3=|11⟩
        if self.n_qubits == 2:
            phi_plus = np.zeros(4, dtype=complex)
            phi_plus[0] = 1.0 / np.sqrt(2)
            phi_plus[3] = 1.0 / np.sqrt(2)
        return np.outer(phi_plus, phi_plus.conj())

    def run(
        self,
        rho0: CMatrix,
        H_ctrl: CMatrix,
        dt: float = 0.01,
        n_steps: int = 1000,
        record_every: int = 10,
    ) -> List[CMatrix]:
        """Evolve density matrix under H_ctrl + Lindblad noise.

        Returns list of saved density matrices.
        """
        return evolve_rk4(
            rho0,
            H_ctrl,
            self._ops,
            dt=dt,
            n_steps=n_steps,
            hbar=self.params.hbar,
            record_every=record_every,
        )

    def entanglement_entropy(self, rho: CMatrix, qubit_a: int = 0) -> float:
        """Von Neumann entropy of the reduced state of qubit_a.

        S(ρ_A) = -Tr(ρ_A log ρ_A)
        """
        n = self.n_qubits
        # Reshape to tensor and trace out complement
        d = self.dim
        rho_tensor = rho.reshape([2] * n + [2] * n)
        # Keep qubit_a, trace others
        # Build index lists for np.einsum
        idx_in = list(range(n))
        idx_out = list(range(n, 2 * n))
        # Partial trace: sum over all qubits except qubit_a
        traced = rho
        axes_to_trace = [k for k in range(n) if k != qubit_a]
        # Use reshape + einsum approach
        rho_A = _partial_trace(rho, n, [qubit_a])
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log(eigvals)))


def _partial_trace(rho: CMatrix, n_qubits: int, keep: List[int]) -> CMatrix:
    """Partial trace: keep only the qubits in `keep`."""
    d = 2 ** n_qubits
    d_keep = 2 ** len(keep)
    d_trace = d // d_keep

    # Reshape to (2,)*n x (2,)*n tensor
    tensor = rho.reshape([2] * (2 * n_qubits))

    trace_qubits = sorted(set(range(n_qubits)) - set(keep))

    # Build einsum string: trace over qubits not in keep
    # Input indices: a0 a1 … a_{n-1} b0 b1 … b_{n-1}
    # Output: kept row, kept col
    in_chars = [chr(ord('a') + i) for i in range(n_qubits)]
    out_chars = [chr(ord('A') + i) for i in range(n_qubits)]

    # For traced qubits: in_char == out_char (set equal for contraction)
    for tq in trace_qubits:
        out_chars[tq] = in_chars[tq]

    lhs = ''.join(in_chars) + ''.join(out_chars)
    # RHS: kept in, kept out
    rhs = ''.join(in_chars[k] for k in keep) + ''.join(
        out_chars[k] for k in keep if out_chars[k] != in_chars[k]
        # but we want the kept out chars to be different
    )
    # Simpler manual loop for reliability
    keep_set = set(keep)
    trace_set = set(trace_qubits)
    trace_list = sorted(trace_set)

    result = tensor
    # Trace one qubit at a time (from high index to low to preserve indexing)
    offset = 0
    for tq in sorted(trace_list, reverse=True):
        axis = tq - offset  # axis shifts as we trace
        # np.trace over axes (axis, axis + remaining_n)
        remaining_n = n_qubits - offset
        ax1 = axis
        ax2 = axis + remaining_n
        result = np.trace(result, axis1=ax1, axis2=ax2)
        offset += 1

    # Result shape: (2,)*len(keep) x (2,)*len(keep) — need to reshape
    n_keep = len(keep)
    result = result.reshape(2 ** n_keep, 2 ** n_keep)
    return result

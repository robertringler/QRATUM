r"""Tensor-network MPS representation for N-qubit states — GAP 5 closure.

Motivation
----------
Full density-matrix simulation of N qubits requires O(4^N) memory, becoming
intractable around N ≈ 10 on a standard workstation.  Matrix Product States
(MPS) provide an efficient representation for states with limited entanglement:

    |ψ⟩ = Σ_{σ₁…σ_N} A[1]^{σ₁} A[2]^{σ₂} … A[N]^{σ_N} |σ₁…σ_N⟩

where each A[k]^{σ_k} is a χ_{k-1} × χ_k matrix (bond dimension χ).

Memory: O(N · d · χ²)  vs  O(d^N)   (d=2 for qubits)
Exact for χ = 2^{N/2} (full entanglement); approximate for smaller χ.

Tensor storage convention
--------------------------
tensors[k].shape = (d=2, chi_left, chi_right)
tensors[k][sigma, alpha, beta] = A^sigma_{alpha, beta}

State vector ordering: psi[σ₁·2^{N-1} + σ₂·2^{N-2} + … + σ_N]

Left-canonical form: Σ_{σ, α} (A^σ)^*_{α,β} (A^σ)_{α,γ} = δ_{β,γ}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]
RVec = NDArray[np.float64]


# ---------------------------------------------------------------------------
# MPS data structure
# ---------------------------------------------------------------------------

class MPSState:
    r"""Matrix Product State for a pure N-qubit state.

    Parameters
    ----------
    tensors  : list of np.ndarray, tensors[k].shape = (d=2, chi_left, chi_right)
    chi_max  : maximum bond dimension for truncation
    """

    def __init__(self, tensors: List[CMatrix], chi_max: int = 64) -> None:
        self.tensors = [t.copy() for t in tensors]
        self.n_qubits = len(tensors)
        self.chi_max = chi_max

    @property
    def n(self) -> int:
        return self.n_qubits

    def copy(self) -> "MPSState":
        return MPSState([t.copy() for t in self.tensors], self.chi_max)

    def norm(self) -> float:
        """Compute ‖|ψ⟩‖ via explicit state vector (only for small N)."""
        psi = mps_to_vector(self)
        return float(np.linalg.norm(psi))

    def normalize(self) -> None:
        """Normalize the MPS in-place by scaling the last tensor."""
        nrm = self.norm()
        if nrm > 1e-15:
            self.tensors[-1] = self.tensors[-1] / nrm


# ---------------------------------------------------------------------------
# MPS constructors
# ---------------------------------------------------------------------------

def mps_ground_state(n_qubits: int, chi_max: int = 64) -> MPSState:
    """Return |00…0⟩ as an MPS with bond dimension 1 (product state)."""
    tensors = []
    for _ in range(n_qubits):
        t = np.zeros((2, 1, 1), dtype=complex)
        t[0, 0, 0] = 1.0   # |0⟩
        tensors.append(t)
    return MPSState(tensors, chi_max)


def mps_product_state(state_list: List[int], chi_max: int = 64) -> MPSState:
    """Return |s₁ s₂ … s_N⟩ for s_k ∈ {0, 1}."""
    tensors = []
    for s in state_list:
        t = np.zeros((2, 1, 1), dtype=complex)
        t[s, 0, 0] = 1.0
        tensors.append(t)
    return MPSState(tensors, chi_max)


def mps_bell_state(n_qubits: int, chi_max: int = 64) -> MPSState:
    r"""Return |Φ+⟩ = (|00⟩+|11⟩)/√2 on qubits 0,1; rest in |0⟩.

    Tensor construction:
      A[0]: shape (2, 1, 2)  — A[0][σ₁, 0, α]
        A[0][0, 0, 0] = 1/√2  (|0⟩ → bond 0)
        A[0][1, 0, 1] = 1/√2  (|1⟩ → bond 1)
      A[1]: shape (2, 2, 1)  — A[1][σ₂, α, 0]
        A[1][0, 0, 0] = 1     (bond 0 → |0⟩)
        A[1][1, 1, 0] = 1     (bond 1 → |1⟩)
    """
    if n_qubits < 2:
        raise ValueError("Bell state requires n_qubits ≥ 2")
    tensors = []

    A0 = np.zeros((2, 1, 2), dtype=complex)
    A0[0, 0, 0] = 1.0 / np.sqrt(2)
    A0[1, 0, 1] = 1.0 / np.sqrt(2)
    tensors.append(A0)

    A1 = np.zeros((2, 2, 1), dtype=complex)
    A1[0, 0, 0] = 1.0
    A1[1, 1, 0] = 1.0
    tensors.append(A1)

    for _ in range(2, n_qubits):
        t = np.zeros((2, 1, 1), dtype=complex)
        t[0, 0, 0] = 1.0
        tensors.append(t)

    return MPSState(tensors, chi_max)


def mps_random(
    n_qubits: int,
    chi_max: int = 4,
    seed: Optional[int] = None,
) -> MPSState:
    """Construct a random normalized MPS with bond dimension ≤ chi_max."""
    rng = np.random.default_rng(seed)

    # Build a random state vector and decompose to MPS
    psi = rng.standard_normal(2 ** n_qubits) + 1j * rng.standard_normal(2 ** n_qubits)
    psi /= np.linalg.norm(psi)
    return _vector_to_mps(psi, n_qubits, chi_max)


# ---------------------------------------------------------------------------
# State-vector reconstruction (exact, O(2^N · N · χ²))
# ---------------------------------------------------------------------------

def mps_to_vector(mps: MPSState) -> CMatrix:
    """Contract MPS tensors to the full state vector of shape (2^N,).

    Algorithm: sequential left-to-right contraction.

    result[σ₁…σ_k, α] ← result[σ₁…σ_{k-1}, β] × A[k][σ_k, β, α]

    At the end result has shape (2^N, 1); we return result[:, 0].

    Warning: exponential in N — only use for N ≤ 20.
    """
    n = mps.n_qubits

    # result[σ₁, α] = A[0][σ₁, 0, α]  (squeeze chi_L = 1)
    result = mps.tensors[0][:, 0, :].copy()   # (d=2, chi_R_0)

    for k in range(1, n):
        A = mps.tensors[k]   # (d=2, chi_L, chi_R)
        d_cur = result.shape[0]    # 2^k
        chi_R = A.shape[2]

        # new_result[σ₁…σ_{k-1}, σ_k, β] = result[σ₁…σ_{k-1}, α] · A[σ_k, α, β]
        # For σ_k ∈ {0, 1}: new[:, σ_k, :] = result @ A[σ_k, :, :]
        # Stack along σ_k axis then reshape to (2^{k+1}, chi_R)
        new_result = np.stack(
            [result @ A[s] for s in range(2)], axis=1
        )  # (d_cur, 2, chi_R)
        result = new_result.reshape(d_cur * 2, chi_R)

    return result[:, 0]


# ---------------------------------------------------------------------------
# Convert state vector to MPS via sequential SVDs
# ---------------------------------------------------------------------------

def _vector_to_mps(psi: CMatrix, n_qubits: int, chi_max: int = 64) -> MPSState:
    """Convert a full state vector to left-canonical MPS via sequential SVDs.

    At each step the (chi_left × 2) × dim_right matrix is SVD-decomposed.
    The left singular vectors form the next MPS tensor, and the diagonal
    matrix times right singular vectors form the new ``remaining`` matrix.
    """
    d = 2
    tensors = []
    chi_left = 1
    remaining = psi.astype(complex).copy()   # shape (2^N,)

    for k in range(n_qubits - 1):
        dim_right = 2 ** (n_qubits - k - 1)
        # Reshape: rows index (alpha_left, sigma_k), cols index remaining qubits
        mat = remaining.reshape(chi_left * d, dim_right)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        chi_new = max(1, min(chi_max, len(S)))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        # U: (chi_left * d, chi_new) → (d, chi_left, chi_new) via reshape + transpose
        # U.reshape(chi_left, d, chi_new)[α_L, σ, α_R] = U[α_L * d + σ, α_R]
        # Transpose to (d, chi_left, chi_new): tensor[σ, α_L, α_R] ✓
        tensors.append(U.reshape(chi_left, d, chi_new).transpose(1, 0, 2))

        remaining = np.diag(S) @ Vh   # (chi_new, dim_right)
        chi_left = chi_new

    # Last tensor: remaining has shape (chi_left, d)
    # We want tensor[σ, α, 0] = remaining[α, σ]
    # remaining.reshape(chi_left, d, 1).transpose(1, 0, 2) → (d, chi_left, 1)
    tensors.append(remaining.reshape(chi_left, d, 1).transpose(1, 0, 2))

    return MPSState(tensors, chi_max)


# ---------------------------------------------------------------------------
# Left canonicalization via sequential QR decompositions
# ---------------------------------------------------------------------------

def _left_canonicalize(mps: MPSState) -> None:
    """Convert MPS to left-canonical form in place via QR decompositions.

    For each site k from 0 to N−2:
      1. Reshape A[k] to (d * chi_L, chi_R) and QR-decompose.
      2. Store Q.reshape(d, chi_L, chi_new) as the new tensor[k].
      3. Absorb R into tensor[k+1].
    """
    n = mps.n_qubits
    for k in range(n - 1):
        A = mps.tensors[k]    # (d, chi_L, chi_R)
        d, chi_L, chi_R = A.shape
        mat = A.reshape(d * chi_L, chi_R)   # rows: (σ, α_L), cols: α_R
        Q, R = np.linalg.qr(mat)            # Q: (d*chi_L, chi_new), R: (chi_new, chi_R)
        chi_new = Q.shape[1]
        mps.tensors[k] = Q.reshape(d, chi_L, chi_new)

        B = mps.tensors[k + 1]   # (d, chi_R, chi_R2) where chi_R = chi_new now expected
        # Absorb R: new_B[σ, i, j] = Σ_m R[i, m] * B[σ, m, j]
        # = R @ B[σ] for each σ
        mps.tensors[k + 1] = np.stack(
            [R @ B[s] for s in range(d)], axis=0
        )   # (d, chi_new, chi_R2)


# ---------------------------------------------------------------------------
# Local gate application
# ---------------------------------------------------------------------------

def apply_single_qubit_gate(mps: MPSState, gate: CMatrix, site: int) -> MPSState:
    """Apply a 2×2 unitary gate to qubit `site`.

    Returns a new MPS (original not modified).
    """
    mps_new = mps.copy()
    A = mps_new.tensors[site]    # (d, chi_L, chi_R)
    # A_new[σ', α, β] = Σ_σ gate[σ', σ] A[σ, α, β]
    # = np.tensordot(gate, A, axes=([1], [0]))  → (d, d, chi_L, chi_R)[σ', σ, α, β]
    # then take diagonal over σ: not needed; just contract:
    mps_new.tensors[site] = np.stack(
        [sum(gate[s_new, s] * A[s] for s in range(2)) for s_new in range(2)],
        axis=0
    )   # (d, chi_L, chi_R)
    return mps_new


def apply_two_qubit_gate(
    mps: MPSState,
    gate: CMatrix,
    site: int,
    chi_max: Optional[int] = None,
    eps_svd: float = 1e-10,
) -> MPSState:
    r"""Apply a 4×4 two-qubit gate to (site, site+1) via SVD truncation.

    Steps:
    1. Form Θ[σ_k, σ_{k+1}, α_L, α_R] = Σ_{γ} A[k][σ_k, α_L, γ] A[k+1][σ_{k+1}, γ, α_R]
    2. Apply gate: Θ' = gate · Θ  (contract over physical indices)
    3. Reshape Θ'[σ_k α_L, σ_{k+1} α_R] and SVD with truncation.
    4. Store updated tensors A[k], A[k+1].
    """
    if chi_max is None:
        chi_max = mps.chi_max

    mps_new = mps.copy()
    A = mps_new.tensors[site]          # (d, chi_L, chi_M)
    B = mps_new.tensors[site + 1]      # (d, chi_M, chi_R)

    d = 2
    chi_L = A.shape[1]
    chi_M = A.shape[2]
    chi_R = B.shape[2]

    # Step 1: Θ[σ_k, σ_{k+1}, α_L, α_R] = Σ_γ A[σ_k, α_L, γ] B[σ_{k+1}, γ, α_R]
    # For each (σ_k, σ_{k+1}): Θ_slice = A[σ_k] @ B[σ_{k+1}]  (chi_L, chi_R)
    Theta = np.zeros((d, d, chi_L, chi_R), dtype=complex)
    for sl in range(d):
        for sr in range(d):
            Theta[sl, sr] = A[sl] @ B[sr]

    # Step 2: Apply gate (reshape as (d,d,d,d) tensor)
    gate4 = gate.reshape(d, d, d, d)
    # Theta_new[σ'_k, σ'_{k+1}, α_L, α_R] = Σ_{σ_k, σ_{k+1}} gate4[σ'_k, σ'_{k+1}, σ_k, σ_{k+1}] Theta[σ_k, σ_{k+1}, α_L, α_R]
    Theta_new = np.zeros((d, d, chi_L, chi_R), dtype=complex)
    for sl_new in range(d):
        for sr_new in range(d):
            for sl in range(d):
                for sr in range(d):
                    Theta_new[sl_new, sr_new] += gate4[sl_new, sr_new, sl, sr] * Theta[sl, sr]

    # Step 3: Reshape to (d * chi_L, d * chi_R) for SVD
    # Row index: (σ'_k, α_L) — σ'_k is slow, α_L is fast
    # Col index: (σ'_{k+1}, α_R) — σ'_{k+1} is slow, α_R is fast
    mat = Theta_new.transpose(0, 2, 1, 3).reshape(d * chi_L, d * chi_R)
    # mat[σ'_k * chi_L + α_L, σ'_{k+1} * chi_R + α_R] = Theta_new[σ'_k, σ'_{k+1}, α_L, α_R] ✓

    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    chi_new = max(1, min(chi_max, int(np.sum(S > eps_svd))))

    U = U[:, :chi_new]
    S_sqrt = np.sqrt(S[:chi_new])
    Vh = Vh[:chi_new, :]

    # Absorb √S into both sides
    U_s = U * S_sqrt[np.newaxis, :]    # (d * chi_L, chi_new)
    V_s = (Vh.T * S_sqrt[np.newaxis, :]).T   # (chi_new, d * chi_R)

    # Reshape back:
    # A_new: (d * chi_L, chi_new) → (d, chi_L, chi_new)
    mps_new.tensors[site] = U_s.reshape(d, chi_L, chi_new)
    # B_new: (chi_new, d * chi_R) → (d, chi_new, chi_R)
    # V_s[i, σ'_{k+1} * chi_R + α_R] → reshape (chi_new, d, chi_R) → transpose (d, chi_new, chi_R)
    mps_new.tensors[site + 1] = V_s.reshape(chi_new, d, chi_R).transpose(1, 0, 2)

    return mps_new


# ---------------------------------------------------------------------------
# Entanglement entropy from Schmidt spectrum
# ---------------------------------------------------------------------------

def entanglement_entropy_at_bond(mps: MPSState, bond: int) -> float:
    r"""Von Neumann entropy S = −Σ_α λ_α² log λ_α² at bond `bond`.

    Reconstructs the full state vector and performs an exact Schmidt
    decomposition at the (bond | bond+1) bipartition.

    Only tractable for N ≤ 20.
    """
    n = mps.n_qubits
    if bond < 0 or bond >= n - 1:
        return 0.0

    psi = mps_to_vector(mps)
    d_left = 2 ** (bond + 1)
    d_right = 2 ** (n - bond - 1)
    mat = psi.reshape(d_left, d_right)
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    S2 = (S ** 2).real
    S2 = S2[S2 > 1e-15]
    return float(-np.sum(S2 * np.log(S2))) if len(S2) > 0 else 0.0


# ---------------------------------------------------------------------------
# Inner product and fidelity
# ---------------------------------------------------------------------------

def mps_overlap(mps1: MPSState, mps2: MPSState) -> complex:
    r"""Compute ⟨ψ₁|ψ₂⟩ via sequential transfer-matrix contraction.

    T[α, β] = Σ_{σ} A1*[σ, :, α] · A2[σ, :, β]  (updated site by site)
    """
    n = mps1.n_qubits
    # T: (chi1_L, chi2_L)
    T = np.ones((1, 1), dtype=complex)

    for k in range(n):
        A1 = mps1.tensors[k]   # (d, chi1_L, chi1_R)
        A2 = mps2.tensors[k]   # (d, chi2_L, chi2_R)
        chi1_R = A1.shape[2]
        chi2_R = A2.shape[2]

        # new_T[i, j] = Σ_{σ, α, β} A1*[σ, α, i] T[α, β] A2[σ, β, j]
        # Step 1: TA2[σ, α, j] = Σ_β T[α, β] A2[σ, β, j] = T @ A2[σ]
        # Step 2: new_T[i, j] = Σ_{σ, α} A1*[σ, α, i] TA2[σ, α, j]
        #                     = Σ_σ A1[σ].conj().T @ TA2[σ]
        new_T = np.zeros((chi1_R, chi2_R), dtype=complex)
        for s in range(2):
            TA2_s = T @ A2[s]                 # (chi1_L, chi2_R)
            new_T += A1[s].conj().T @ TA2_s   # (chi1_R, chi1_L) @ (chi1_L, chi2_R) = (chi1_R, chi2_R)

        T = new_T

    return complex(T[0, 0])


def mps_fidelity(mps1: MPSState, mps2: MPSState) -> float:
    """Fidelity |⟨ψ₁|ψ₂⟩|²."""
    return float(abs(mps_overlap(mps1, mps2)) ** 2)


# ---------------------------------------------------------------------------
# Reduced density matrix
# ---------------------------------------------------------------------------

def mps_reduced_density_matrix(mps: MPSState, keep: List[int]) -> CMatrix:
    r"""Compute ρ_A = Tr_B(|ψ⟩⟨ψ|) by reconstructing the full state vector.

    Only tractable for N ≤ 20.
    """
    psi = mps_to_vector(mps)
    n = mps.n_qubits

    keep_sorted = sorted(keep)
    trace_qubits = [k for k in range(n) if k not in keep_sorted]
    perm = keep_sorted + trace_qubits

    n_keep = len(keep_sorted)
    d_keep = 2 ** n_keep
    d_trace = 2 ** (n - n_keep)

    psi_tensor = psi.reshape([2] * n)
    psi_reordered = psi_tensor.transpose(perm).reshape(d_keep, d_trace)

    return psi_reordered @ psi_reordered.conj().T


# ---------------------------------------------------------------------------
# Density matrix from MPS
# ---------------------------------------------------------------------------

def mps_to_density_matrix(mps: MPSState) -> CMatrix:
    """Compute ρ = |ψ⟩⟨ψ| from MPS (only for small N)."""
    psi = mps_to_vector(mps)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Trotter-Suzuki Hamiltonian evolution (TEBD)
# ---------------------------------------------------------------------------

def _make_two_site_gate(H_pair: CMatrix, dt: float) -> CMatrix:
    r"""Compute two-site gate exp(−i H_pair dt) via matrix exponentiation."""
    from scipy.linalg import expm
    return expm(-1j * dt * H_pair).astype(complex)


def mps_evolve_trotter(
    mps: MPSState,
    H_pairs: List[Tuple[int, CMatrix]],
    dt: float,
    n_steps: int,
    chi_max: int = 64,
    record_every: int = 1,
) -> List[MPSState]:
    r"""Time-evolve MPS under H = Σ_{⟨ij⟩} H_pairs via first-order Trotter.

    Parameters
    ----------
    mps        : initial MPS (copied, not modified in place)
    H_pairs    : list of (site_index, 4×4 Hamiltonian) pairs
    dt         : time step
    n_steps    : number of Trotter steps
    chi_max    : bond dimension truncation
    record_every: save every this many steps

    Returns
    -------
    list of MPS snapshots (includes initial state)
    """
    current = mps.copy()
    trajectory = [current.copy()]
    gates = [(site, _make_two_site_gate(H, dt)) for site, H in H_pairs]

    for step in range(1, n_steps + 1):
        for site, gate in gates:
            current = apply_two_qubit_gate(current, gate, site, chi_max=chi_max)
        current.normalize()
        if step % record_every == 0:
            trajectory.append(current.copy())

    return trajectory


# ---------------------------------------------------------------------------
# Quantum-trajectory Lindblad evolution (MCWF)
# ---------------------------------------------------------------------------

def mps_lindblad_trajectory(
    mps0: MPSState,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    dt: float,
    n_steps: int,
    chi_max: int = 64,
    seed: Optional[int] = None,
) -> List[MPSState]:
    """Single quantum trajectory via Monte Carlo Wave Function (MCWF) method.

    Algorithm per step:
    1. Non-Hermitian effective Hamiltonian: H_eff = H − i/2 Σ_k L_k† L_k
    2. Deterministic evolution: |ψ'⟩ = exp(−i H_eff dt)|ψ⟩
    3. Total jump probability: dp = Σ_k dt ‖L_k|ψ⟩‖²
    4. If r < dp: apply random jump L_k (chosen proportionally) and renorm.
       Else: renormalize |ψ'⟩.

    The average over many trajectories gives ρ(t) = 𝔼[|ψ(t)⟩⟨ψ(t)|].

    Returns list of MPS snapshots (length n_steps + 1).
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    n = mps0.n_qubits

    # Build effective Hamiltonian
    H_eff = H.astype(complex) - 0.5j * sum(
        L.conj().T @ L for L in lindblad_ops
    )
    exp_H_eff = expm(-1j * dt * H_eff)

    current = mps0.copy()
    trajectory = [current.copy()]

    for _ in range(n_steps):
        psi = mps_to_vector(current)

        # Deterministic non-Hermitian step
        psi_new = exp_H_eff @ psi

        # Jump probabilities
        dp_k = np.array([
            dt * float(np.real(psi.conj() @ (L.conj().T @ L) @ psi))
            for L in lindblad_ops
        ])
        dp_total = float(np.clip(np.sum(dp_k), 0.0, 1.0))

        r = rng.uniform()
        if r < dp_total and dp_total > 0:
            probs = np.clip(dp_k, 0.0, None)
            probs /= probs.sum()
            k_jump = int(rng.choice(len(lindblad_ops), p=probs))
            psi_j = lindblad_ops[k_jump] @ psi
            nrm = float(np.linalg.norm(psi_j))
            if nrm > 1e-15:
                psi_j /= nrm
            current = _vector_to_mps(psi_j, n, chi_max)
        else:
            nrm = float(np.linalg.norm(psi_new))
            if nrm > 1e-15:
                psi_new /= nrm
            current = _vector_to_mps(psi_new, n, chi_max)

        trajectory.append(current.copy())

    return trajectory


# ---------------------------------------------------------------------------
# Scalability comparison helper
# ---------------------------------------------------------------------------

def compare_exact_vs_mps(
    n_qubits: int,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    rho0: CMatrix,
    dt: float = 0.01,
    n_steps: int = 50,
    chi_max: int = 16,
    n_trajectories: int = 50,
    seed: int = 42,
) -> dict:
    """Compare exact Lindblad vs MPS quantum-trajectory density matrix.

    Only valid for N ≤ 6 (where exact density-matrix simulation is tractable).

    Returns
    -------
    dict with keys:
      fidelity_vs_exact : overlap between MCWF average and exact
      mps_memory_bytes  : bytes used by initial MPS
      exact_memory_bytes: bytes for exact density matrix
      compression_ratio : exact / mps
    """
    from quasim.ciir.multi_qubit.quantum.density_matrix import evolve_rk4

    # Exact evolution
    rho_traj_exact = evolve_rk4(rho0, H.astype(complex), lindblad_ops, dt, n_steps)
    rho_final_exact = rho_traj_exact[-1]

    # Initial pure state: dominant eigenvector of rho0
    eigvals, eigvecs = np.linalg.eigh(rho0)
    psi0 = eigvecs[:, -1].astype(complex)
    psi0 /= np.linalg.norm(psi0)
    mps0 = _vector_to_mps(psi0, n_qubits, chi_max)

    # Average over trajectories
    rho_avg = np.zeros_like(rho0)
    for s in range(n_trajectories):
        traj = mps_lindblad_trajectory(
            mps0, H, lindblad_ops, dt, n_steps, chi_max=chi_max, seed=seed + s
        )
        psi_f = mps_to_vector(traj[-1])
        rho_avg += np.outer(psi_f, psi_f.conj())
    rho_avg /= n_trajectories

    fidelity = float(np.real(np.trace(rho_avg @ rho_final_exact)))
    exact_bytes = rho0.nbytes
    mps_bytes = sum(t.nbytes for t in mps0.tensors)

    return {
        "fidelity_vs_exact": fidelity,
        "mps_memory_bytes": mps_bytes,
        "exact_memory_bytes": exact_bytes,
        "compression_ratio": exact_bytes / max(1, mps_bytes),
        "rho_mps_avg": rho_avg,
        "rho_exact": rho_final_exact,
    }

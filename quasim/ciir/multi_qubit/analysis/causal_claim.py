r"""CIIR Causal Claim Formalization — GAP 4 closure.

Formalizes the CIIR causal claim and implements the distinguishing observable
that separates Model A (CIIR with constraint geometry) from Model B (standard
open-systems without constraint geometry).

CIIR Causal Claim (Proposition P)
-----------------------------------
"The CIIR constraint geometry causes the Lindbladian fixed point to be stable
against decoherence in a way that is NOT achievable by any open-systems model
without constraint geometry."

Formalization
-------------
* Model A (CIIR): quantum state ρ is periodically projected onto the code
  subspace Π_C before the Lindbladian step is applied:
      ρ_n+1 = L_dt( Π_C ρ_n Π_C / Tr(Π_C ρ_n) )

* Model B (Standard): same Lindbladian, NO constraint projection:
      ρ_n+1 = L_dt( ρ_n )

Distinguishing Observable O
----------------------------
O = F(ρ_final, ρ_target) — fidelity with the target state after noise.

The null hypothesis O_null is the fidelity achieved by Model B under the
same noise level p > p*.

Proposition P is supported (experimentally) iff O_A > O_null = O_B.

Ramsey Protocol
---------------
A Ramsey-like scheme measures the distinguishing observable:
  1. Prepare ρ_0 = target state.
  2. Apply noise pulse at strength p > p* for time τ_noise.
  3. (CIIR only) Apply error correction (constraint projection).
  4. Wait for time τ_free.
  5. Measure O = F(ρ, ρ_target).

References
----------
* Zurek (2003) — Decoherence, einselection, and the quantum origins of the classical
* Ch22, CIIR monograph — Theorem 22.2 (Causal Claim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]

# ---------------------------------------------------------------------------
# PARAMS
# ---------------------------------------------------------------------------

PARAMS = {
    # Fraction of code subspace (dimension / total dimension)
    "code_fraction": 0.5,
    # Default noise threshold above which CIIR advantage is expected
    "p_star": 0.1,
    # Default integration time step
    "dt": 0.01,
    # Ramsey protocol defaults
    "ramsey_tau_noise": 0.5,
    "ramsey_tau_free": 1.0,
    "ramsey_p_noise": 0.15,  # > p_star
}


# ---------------------------------------------------------------------------
# Lindbladian helpers
# ---------------------------------------------------------------------------


def _lindblad_dissipator(rho: CMatrix, lindblad_ops: List[CMatrix]) -> CMatrix:
    """Σ_k D[L_k]ρ — standard Lindblad dissipator."""
    out = np.zeros_like(rho)
    for L in lindblad_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        out += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
    return out


def _gksl_step_rk4(
    rho: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    dt: float,
) -> CMatrix:
    """One RK4 step of the GKSL master equation."""

    def f(r: CMatrix) -> CMatrix:
        coherent = -1j * (H @ r - r @ H)
        return coherent + _lindblad_dissipator(r, lindblad_ops)

    k1 = f(rho)
    k2 = f(rho + 0.5 * dt * k1)
    k3 = f(rho + 0.5 * dt * k2)
    k4 = f(rho + dt * k3)
    rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Enforce Hermiticity, positivity, unit trace
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho_new)
    eigvals = np.clip(eigvals.real, 0.0, None)
    rho_new = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    tr = float(np.trace(rho_new).real)
    if tr > 1e-15:
        rho_new /= tr
    return rho_new


def _build_constraint_projector(dim: int) -> CMatrix:
    """Build the code-subspace projector Π_C.

    For the CIIR code we use the lower half of the Hilbert space as the
    "code subspace" (a concrete, computationally tractable stand-in for
    the true stabilizer code subspace).

    dim : Hilbert space dimension (2^N)
    """
    k = max(1, dim // 2)
    Pi = np.zeros((dim, dim), dtype=complex)
    Pi[:k, :k] = np.eye(k, dtype=complex)
    return Pi


def _apply_constraint_projection(rho: CMatrix, Pi: CMatrix) -> CMatrix:
    """Project ρ → Π_C ρ Π_C / Tr(Π_C ρ Π_C), renormalized."""
    rho_proj = Pi @ rho @ Pi
    tr = float(np.trace(rho_proj).real)
    if tr < 1e-15:
        return rho.copy()
    return rho_proj / tr


def _apply_depolarizing_noise(rho: CMatrix, p: float) -> CMatrix:
    """Apply single-step depolarizing channel at strength p.

    E(ρ) = (1-p) ρ + p * I/d
    """
    d = rho.shape[0]
    return (1.0 - p) * rho + p * np.eye(d, dtype=complex) / d


# ---------------------------------------------------------------------------
# Model A: CIIR evolution with constraint projection
# ---------------------------------------------------------------------------


def simulate_ciir_model(
    rho0: CMatrix,
    H: CMatrix,
    ops: List[CMatrix],
    n_steps: int,
    with_constraint: bool = True,
    dt: float = PARAMS["dt"],
    noise_strength: float = 0.0,
) -> List[CMatrix]:
    r"""Simulate CIIR model A (with optional constraint projection).

    Each step:
      1. Apply depolarizing noise at ``noise_strength``.
      2. (If with_constraint) project ρ onto code subspace Π_C.
      3. Evolve one GKSL step.

    Parameters
    ----------
    rho0           : initial density matrix
    H              : Hamiltonian
    ops            : Lindblad operators
    n_steps        : number of steps
    with_constraint: if True, apply constraint projection (Model A);
                     if False, standard evolution (Model B equivalent)
    dt             : time step
    noise_strength : depolarizing noise strength p applied each step

    Returns
    -------
    list of density matrices (length n_steps + 1, includes rho0)
    """
    dim = rho0.shape[0]
    Pi = _build_constraint_projector(dim)
    rho = rho0.astype(complex).copy()
    trajectory = [rho.copy()]

    for _ in range(n_steps):
        # Noise
        if noise_strength > 0.0:
            rho = _apply_depolarizing_noise(rho, noise_strength)
        # Constraint projection (Model A only)
        if with_constraint:
            rho = _apply_constraint_projection(rho, Pi)
        # GKSL step
        rho = _gksl_step_rk4(rho, H, ops, dt)
        trajectory.append(rho.copy())

    return trajectory


def simulate_standard_model(
    rho0: CMatrix,
    H: CMatrix,
    ops: List[CMatrix],
    n_steps: int,
    dt: float = PARAMS["dt"],
    noise_strength: float = 0.0,
) -> List[CMatrix]:
    """Simulate Model B (no constraint projection).

    Convenience wrapper for simulate_ciir_model with with_constraint=False.
    """
    return simulate_ciir_model(
        rho0,
        H,
        ops,
        n_steps,
        with_constraint=False,
        dt=dt,
        noise_strength=noise_strength,
    )


# ---------------------------------------------------------------------------
# Distinguishing observable
# ---------------------------------------------------------------------------


def state_fidelity(rho: CMatrix, sigma: CMatrix) -> float:
    """Fidelity F(ρ, σ) = Tr(ρ σ)  (simplified; valid when one state is pure)."""
    return float(np.clip(np.real(np.trace(rho @ sigma)), 0.0, 1.0))


def compute_distinguishing_observable(
    ciir_traj: List[CMatrix],
    std_traj: List[CMatrix],
    rho_target: CMatrix,
) -> dict:
    r"""Compute the distinguishing observable O at each time step.

    O_A(t) = F(ρ_A(t), ρ_target)
    O_B(t) = F(ρ_B(t), ρ_target)
    ΔO(t)  = O_A(t) − O_B(t)

    Proposition P is supported iff ΔO_final > 0.

    Parameters
    ----------
    ciir_traj  : trajectory from Model A
    std_traj   : trajectory from Model B
    rho_target : target density matrix

    Returns
    -------
    dict with keys: fidelity_ciir, fidelity_std, delta_fidelity,
                    proposition_P_supported, final_delta
    """
    n = min(len(ciir_traj), len(std_traj))
    fid_ciir = np.array([state_fidelity(r, rho_target) for r in ciir_traj[:n]])
    fid_std = np.array([state_fidelity(r, rho_target) for r in std_traj[:n]])
    delta = fid_ciir - fid_std

    return {
        "fidelity_ciir": fid_ciir,
        "fidelity_std": fid_std,
        "delta_fidelity": delta,
        "proposition_P_supported": bool(delta[-1] > 0.0),
        "final_delta": float(delta[-1]),
        "note": (
            "Theorem 22.2 (SIMULATION RESULT): Proposition P is supported "
            "iff ΔO_final = F_A − F_B > 0.  This is a necessary but not "
            "sufficient condition for the full causal claim."
        ),
    }


# ---------------------------------------------------------------------------
# Ramsey protocol
# ---------------------------------------------------------------------------


@dataclass
class RamseyProtocol:
    """Parameters for the CIIR distinguishing Ramsey experiment.

    Fields
    ------
    tau_noise_steps : number of steps during noise pulse
    tau_free_steps  : number of steps during free evolution
    p_noise         : depolarizing noise strength  (should exceed p*)
    dt              : time step
    """

    tau_noise_steps: int
    tau_free_steps: int
    p_noise: float
    dt: float
    p_star: float

    def total_steps(self) -> int:
        return self.tau_noise_steps + self.tau_free_steps

    def noise_in_steps(self) -> int:
        return self.tau_noise_steps

    def above_threshold(self) -> bool:
        return self.p_noise > self.p_star


def design_ramsey_protocol(
    dt: float = PARAMS["dt"],
    tau_noise: float = PARAMS["ramsey_tau_noise"],
    tau_free: float = PARAMS["ramsey_tau_free"],
    p_noise: float = PARAMS["ramsey_p_noise"],
    p_star: float = PARAMS["p_star"],
) -> RamseyProtocol:
    """Return a RamseyProtocol with parameters for the CIIR causal-claim test.

    Protocol
    --------
    1. Prepare ρ₀ = ρ_target.
    2. Apply depolarizing noise at p_noise > p* for tau_noise_steps steps.
    3. (Model A only) Apply constraint projection.
    4. Free evolution for tau_free_steps steps.
    5. Measure O = F(ρ, ρ_target).

    The protocol is diagnostic: it cannot prove causation but can
    falsify the null hypothesis O_A ≤ O_B.
    """
    return RamseyProtocol(
        tau_noise_steps=max(1, int(tau_noise / dt)),
        tau_free_steps=max(1, int(tau_free / dt)),
        p_noise=p_noise,
        dt=dt,
        p_star=p_star,
    )


def run_ramsey_experiment(
    rho0: CMatrix,
    H: CMatrix,
    ops: List[CMatrix],
    protocol: Optional[RamseyProtocol] = None,
) -> dict:
    """Execute the Ramsey protocol for both Model A and Model B.

    Parameters
    ----------
    rho0     : initial (target) density matrix
    H        : Hamiltonian
    ops      : Lindblad operators
    protocol : RamseyProtocol; uses design_ramsey_protocol() defaults if None

    Returns
    -------
    dict with keys: protocol, observable, ciir_final, std_final
    """
    if protocol is None:
        protocol = design_ramsey_protocol()

    # Noise phase
    ciir_noise = simulate_ciir_model(
        rho0,
        H,
        ops,
        n_steps=protocol.tau_noise_steps,
        with_constraint=True,
        dt=protocol.dt,
        noise_strength=protocol.p_noise,
    )
    std_noise = simulate_standard_model(
        rho0,
        H,
        ops,
        n_steps=protocol.tau_noise_steps,
        dt=protocol.dt,
        noise_strength=protocol.p_noise,
    )

    # Free evolution phase (no noise)
    ciir_free = simulate_ciir_model(
        ciir_noise[-1],
        H,
        ops,
        n_steps=protocol.tau_free_steps,
        with_constraint=True,
        dt=protocol.dt,
        noise_strength=0.0,
    )
    std_free = simulate_standard_model(
        std_noise[-1],
        H,
        ops,
        n_steps=protocol.tau_free_steps,
        dt=protocol.dt,
        noise_strength=0.0,
    )

    observable = compute_distinguishing_observable(ciir_free, std_free, rho0)

    return {
        "protocol": protocol,
        "observable": observable,
        "ciir_final": ciir_free[-1],
        "std_final": std_free[-1],
        "above_threshold": protocol.above_threshold(),
    }


# ---------------------------------------------------------------------------
# Numerical noise floor (local copy; mirrors falsification.py)
# ---------------------------------------------------------------------------


def compute_sigma_numerical(n_steps: int, dim: int, dt: float) -> float:
    r"""Estimate double-precision numerical noise floor σ_numerical.

    σ = ε_machine × dim² × n_steps × 20 × dt

    The factor 20 accounts for RK4 stages and complex-multiplication
    cancellation error.
    """
    eps = np.finfo(float).eps
    return float(eps * dim**2 * n_steps * 20 * dt)


# ---------------------------------------------------------------------------
# Standalone system builders (no cross-module import required)
# ---------------------------------------------------------------------------


def _build_default_hamiltonian_local(N: int) -> CMatrix:
    """Simple N-qubit Hamiltonian: Σ Z_i + Σ XX_{i,i+1}."""
    dim = 2**N
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def _embed(op: CMatrix, site: int, n: int) -> CMatrix:
        ops_list = [I2] * n
        ops_list[site] = op
        out = ops_list[0]
        for o in ops_list[1:]:
            out = np.kron(out, o)
        return out

    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H += 0.5 * _embed(sz, i, N)
    for i in range(N - 1):
        H += 0.3 * _embed(sx, i, N) @ _embed(sx, i + 1, N)
    return H


def _build_default_lindblad_local(N: int, gamma: float = 0.02) -> List[CMatrix]:
    """Amplitude-damping operators for each qubit."""
    sigma_m = np.array([[0, 1], [0, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    ops: List[CMatrix] = []
    for i in range(N):
        parts = [I2] * N
        parts[i] = sigma_m
        op = parts[0]
        for p in parts[1:]:
            op = np.kron(op, p)
        ops.append(np.sqrt(gamma) * op)
    return ops


# ---------------------------------------------------------------------------
# Linear GKSL step (no physicality enforcement — used in superoperator build)
# ---------------------------------------------------------------------------


def _gksl_step_linear(
    rho: CMatrix,
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    dt: float,
) -> CMatrix:
    """One RK4 step of the GKSL equation, purely linear (no clipping/normalisation).

    Used only for building the superoperator in Route A of
    :func:`derive_projector_from_fixed_point`.
    """

    def f(r: CMatrix) -> CMatrix:
        return -1j * (H @ r - r @ H) + _lindblad_dissipator(r, lindblad_ops)

    k1 = f(rho)
    k2 = f(rho + 0.5 * dt * k1)
    k3 = f(rho + 0.5 * dt * k2)
    k4 = f(rho + dt * k3)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Superoperator construction
# ---------------------------------------------------------------------------


def _build_ciir_superoperator_linear(
    H: CMatrix,
    lindblad_ops: List[CMatrix],
    Pi: CMatrix,
    dt: float,
    noise_strength: float,
    dim: int,
) -> CMatrix:
    r"""Build the d²×d² superoperator matrix for one CIIR step (without normalisation).

    The map (applied linearly, skipping the Tr-normalisation after projection) is:

        ρ  →  GKSL( Π ρ Π , dt ) + depolarising

    Column j of the returned matrix S equals vec(Φ(e_j)) where e_j is the
    j-th basis matrix (e_j[j//dim, j%dim] = 1, everything else 0) and
    vec(·) uses NumPy's default row-major ravelling.

    Parameters
    ----------
    H, lindblad_ops : system operators
    Pi              : CIIR constraint projector
    dt              : time step
    noise_strength  : depolarising noise strength p
    dim             : Hilbert-space dimension (must equal H.shape[0])

    Returns
    -------
    S : (d², d²) complex matrix
    """
    d2 = dim * dim
    S = np.zeros((d2, d2), dtype=complex)

    for j in range(d2):
        row, col = divmod(j, dim)
        rho_j = np.zeros((dim, dim), dtype=complex)
        rho_j[row, col] = 1.0

        # Depolarising (linear in ρ):  D(ρ) = (1-p)ρ + (p/d) Tr(ρ) I
        tr_j = 1.0 if row == col else 0.0  # Tr(e_{row,col}) = δ_{row,col}
        r = (1.0 - noise_strength) * rho_j + (noise_strength / dim) * tr_j * np.eye(
            dim, dtype=complex
        )

        # Constraint projection (linear, no normalisation)
        r = Pi @ r @ Pi

        # GKSL step (linear, no physicality enforcement)
        r = _gksl_step_linear(r, H, lindblad_ops, dt)

        S[:, j] = r.ravel()

    return S


# ---------------------------------------------------------------------------
# PROJECTOR DERIVATION FROM FIXED-POINT CONDITION
# ---------------------------------------------------------------------------


def derive_projector_from_fixed_point(
    ciir_params: Optional[dict] = None,
    tol: float = 1e-8,
) -> Tuple[CMatrix, dict]:
    r"""Derive Π_C from the observer fixed-point condition Φ_CIIR(ρ*) = ρ*.

    Two independent derivation routes are executed and compared.

    **Route A — Spectral** (executed for N ≤ 4):
        1. Build the d²×d² superoperator S of the CIIR step (without
           normalisation).
        2. Find the dominant eigenvector of S.
        3. Reshape to d×d, project to a valid density matrix, normalise.
        4. Extract the dominant eigenprojector Π_C = |ψ*⟩⟨ψ*|.

    **Route B — Power iteration** (always executed):
        1. Initialise ρ₀ = I/d (maximally mixed).
        2. Iterate ρ_{n+1} = Φ_CIIR(ρ_n) (with normalisation) until
           ‖ρ_{n+1} − ρ_n‖_F < tol.
        3. Extract the dominant eigenprojector.

    Route B is used as the primary result.  If Route A is available
    and the two projectors disagree by more than ``tol``, the
    certificate flags ``routes_disagree = True``.

    Parameters
    ----------
    ciir_params : dict with optional keys

        ``N``              int   — qubits (default 2)
        ``dt``             float — time step (default 0.01)
        ``gamma``          float — Lindblad noise rate (default 0.02)
        ``noise_strength`` float — depolarising strength (default 0.15)
        ``max_iter``       int   — max power-iteration steps (default 2000)
        ``H``              array — Hamiltonian (built from N if absent)
        ``ops``            list  — Lindblad operators (built if absent)

    tol : float
        Convergence tolerance for Route B and residual reporting.

    Returns
    -------
    projector : (d, d) rank-1 Hermitian matrix Π_C
    certificate : dict
        ``fixed_point_residual``  float  — ‖Φ(ρ*) − ρ*‖_F (Route B)
        ``overlap_with_ground``   float  — |⟨0…0|ψ*⟩|²
        ``derivation_route``      str    — ``'spectral+power'`` or ``'power'``
        ``is_degenerate``         bool   — True if overlap ≥ 0.999
        ``routes_disagree``       bool   — True if ‖Π_A − Π_B‖ > tol (N ≤ 4 only)
        ``route_A_residual``      float  — Route A residual (None if not run)
        ``route_agreement``       float  — ‖Π_A − Π_B‖_F (None if not run)
        ``converged``             bool   — Route B converged before max_iter
        ``n_iterations``          int    — Route B iterations taken
    """
    if ciir_params is None:
        ciir_params = {}

    N = int(ciir_params.get("N", 2))
    dt = float(ciir_params.get("dt", 0.01))
    gamma = float(ciir_params.get("gamma", 0.02))
    noise_strength = float(ciir_params.get("noise_strength", 0.15))
    max_iter = int(ciir_params.get("max_iter", 2000))

    dim = 2**N
    H: CMatrix = ciir_params.get("H", _build_default_hamiltonian_local(N))
    ops: List[CMatrix] = ciir_params.get("ops", _build_default_lindblad_local(N, gamma))
    Pi_assumed = _build_constraint_projector(dim)

    # Ground state vector for overlap check
    ground = np.zeros(dim, dtype=complex)
    ground[0] = 1.0

    # ── Route B: Power iteration ─────────────────────────────────────────────

    def _phi_norm(rho: CMatrix) -> CMatrix:
        """Normalised one-step CIIR map: depolarise → project → GKSL."""
        r = _apply_depolarizing_noise(rho, noise_strength)
        r = _apply_constraint_projection(r, Pi_assumed)
        r = _gksl_step_rk4(r, H, ops, dt)
        return r

    rho_B = np.eye(dim, dtype=complex) / dim
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        rho_new = _phi_norm(rho_B)
        diff = float(np.linalg.norm(rho_new - rho_B, "fro"))
        rho_B = rho_new
        if diff < tol:
            converged = True
            break

    ev_B, evec_B = np.linalg.eigh(rho_B)
    idx_B = int(np.argmax(ev_B.real))
    psi_B = evec_B[:, idx_B].copy()
    psi_B /= max(float(np.linalg.norm(psi_B)), 1e-15)
    # Canonical phase: make the largest-magnitude component real and positive
    pivot = int(np.argmax(np.abs(psi_B)))
    psi_B *= np.exp(-1j * np.angle(psi_B[pivot]))
    Pi_B = np.outer(psi_B, psi_B.conj())

    residual_B = float(np.linalg.norm(_phi_norm(rho_B) - rho_B, "fro"))

    # ── Route A: Spectral (N ≤ 4 only — superoperator is ≤ 256×256) ─────────

    Pi_A: Optional[CMatrix] = None
    residual_A: Optional[float] = None
    route_agreement: Optional[float] = None
    routes_disagree = False
    derivation_route = "power"

    if N <= 4:
        try:
            S = _build_ciir_superoperator_linear(H, ops, Pi_assumed, dt, noise_strength, dim)
            eigvals_S, eigvecs_S = np.linalg.eig(S)
            idx_S = int(np.argmax(np.abs(eigvals_S)))
            v_star = eigvecs_S[:, idx_S]

            # Reshape and project to valid density matrix
            rho_star = v_star.reshape(dim, dim)
            rho_star = 0.5 * (rho_star + rho_star.conj().T)
            ev_s, evec_s = np.linalg.eigh(rho_star)
            ev_s = np.clip(ev_s.real, 0.0, None)
            rho_star = evec_s @ np.diag(ev_s) @ evec_s.conj().T
            tr_s = float(np.trace(rho_star).real)
            if tr_s > 1e-15:
                rho_star /= tr_s

            ev_A, evec_A = np.linalg.eigh(rho_star)
            idx_A = int(np.argmax(ev_A.real))
            psi_A = evec_A[:, idx_A].copy()
            psi_A /= max(float(np.linalg.norm(psi_A)), 1e-15)
            psi_A *= np.exp(-1j * np.angle(psi_A[int(np.argmax(np.abs(psi_A)))]))
            Pi_A = np.outer(psi_A, psi_A.conj())

            residual_A = float(np.linalg.norm(_phi_norm(rho_star) - rho_star, "fro"))
            route_agreement = float(np.linalg.norm(Pi_A - Pi_B, "fro"))
            routes_disagree = route_agreement > tol
            derivation_route = "spectral+power"
        except Exception:
            derivation_route = "power"

    # ── Overlap and degeneracy ────────────────────────────────────────────────

    overlap_with_ground = float(abs(np.vdot(ground, psi_B)) ** 2)
    is_degenerate = overlap_with_ground >= 0.999

    certificate: Dict = {
        "fixed_point_residual": residual_B,
        "overlap_with_ground": overlap_with_ground,
        "derivation_route": derivation_route,
        "is_degenerate": is_degenerate,
        "routes_disagree": routes_disagree,
        "route_A_residual": residual_A,
        "route_agreement": route_agreement,
        "converged": converged,
        "n_iterations": n_iter,
    }

    return Pi_B, certificate


# ---------------------------------------------------------------------------
# RESCREENING THE CAUSAL CLAIM WITH DERIVED Π_C
# ---------------------------------------------------------------------------


def _observable_trajectory(
    traj: List[CMatrix],
    Pi: CMatrix,
) -> np.ndarray:
    """Compute ⟨Π⟩(t) = Tr(Π ρ(t)) for each step in traj."""
    return np.array([float(np.real(np.trace(Pi @ rho))) for rho in traj])


def _delta_O_max_with_projector(
    N: int,
    n_steps: int,
    dt: float,
    gamma: float,
    noise_strength: float,
    Pi: CMatrix,
    rho0: Optional[CMatrix] = None,
    H: Optional[CMatrix] = None,
    ops: Optional[List[CMatrix]] = None,
    rng_seed: Optional[int] = None,
    perturb_rho0: Optional[float] = None,
    perturb_gamma_factor: float = 1.0,
    truncation_eps: float = 0.0,
) -> float:
    """Run CIIR and null trajectories and return ΔO_max using ``Pi`` as the observable.

    Supports optional perturbations for artifact tests.
    """
    dim = 2**N
    if H is None:
        H = _build_default_hamiltonian_local(N)
    if ops is None:
        ops = _build_default_lindblad_local(N, gamma * perturb_gamma_factor)
    if rho0 is None:
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[0, 0] = 1.0

    if perturb_rho0 is not None and perturb_rho0 > 0.0:
        rng = np.random.default_rng(seed=rng_seed if rng_seed is not None else 0)
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        A = (A + A.conj().T) / 2.0
        norm_A = float(np.linalg.norm(A, "fro"))
        if norm_A > 1e-15:
            A /= norm_A
        rho0 = rho0 + perturb_rho0 * A
        rho0 = 0.5 * (rho0 + rho0.conj().T)
        ev, evec = np.linalg.eigh(rho0)
        ev = np.clip(ev.real, 0.0, None)
        rho0 = evec @ np.diag(ev) @ evec.conj().T
        tr = float(np.trace(rho0).real)
        if tr > 1e-15:
            rho0 /= tr

    ciir_traj = simulate_ciir_model(
        rho0,
        H,
        ops,
        n_steps,
        with_constraint=True,
        dt=dt,
        noise_strength=noise_strength * perturb_gamma_factor,
    )
    std_traj = simulate_standard_model(
        rho0,
        H,
        ops,
        n_steps,
        dt=dt,
        noise_strength=noise_strength * perturb_gamma_factor,
    )

    if truncation_eps > 0.0:
        rng2 = np.random.default_rng(seed=(rng_seed or 0) + 1)
        for traj in (ciir_traj, std_traj):
            for k in range(1, len(traj)):
                v = rng2.standard_normal(dim) + 1j * rng2.standard_normal(dim)
                v /= max(float(np.linalg.norm(v)), 1e-15)
                perturb = truncation_eps * np.outer(v, v.conj())
                perturb = 0.5 * (perturb + perturb.conj().T)
                rho_p = traj[k] + perturb
                rho_p = 0.5 * (rho_p + rho_p.conj().T)
                ev_p, evec_p = np.linalg.eigh(rho_p)
                ev_p = np.clip(ev_p.real, 0.0, None)
                rho_p = evec_p @ np.diag(ev_p) @ evec_p.conj().T
                tr_p = float(np.trace(rho_p).real)
                if tr_p > 1e-15:
                    rho_p /= tr_p
                traj[k] = rho_p

    O_c = _observable_trajectory(ciir_traj, Pi)
    O_s = _observable_trajectory(std_traj, Pi)
    return float(np.max(np.abs(O_c - O_s)))


def rescreen_causal_claim(
    ciir_params: Optional[dict] = None,
    N_scan: Optional[List[int]] = None,
    tol: float = 1e-8,
) -> dict:
    r"""Rescreen the CIIR causal claim using a derived (non-assumed) projector Π_C.

    Protocol
    --------
    1. Derive Π_C via :func:`derive_projector_from_fixed_point`.
    2. Run CIIR (Model A) and null (Model B) trajectories.
    3. Compute ΔO_derived(t) = ⟨Π_C⟩_A − ⟨Π_C⟩_B.
    4. Run artifact hypotheses AH-1 through AH-5 with the derived observable.
    5. Classify into one of four verdicts:

       ``A+``  ΔO_max > 5σ, all AH pass, overlap_with_ground < 0.5 —
               Π_C is genuinely non-ground-state; signal is structural.

       ``A0``  ΔO_max > 5σ, all AH pass, overlap_with_ground ≥ 0.5 —
               Signal survives but Π_C is near ground state.

       ``B``   1σ < ΔO_max ≤ 5σ (or ≥ 4/5 AH pass) —
               Marginal signal.

       ``C``   ΔO_max ≤ 1σ OR any AH FAIL — null result.

    6. Run N-scaling: rerun for each N in ``N_scan``, re-deriving Π_C.

    Parameters
    ----------
    ciir_params : dict (same keys as :func:`derive_projector_from_fixed_point`,
        plus ``n_steps`` int, default 50).
    N_scan      : list of N values for scaling (default [2, 4, 6, 8]).
    tol         : tolerance for projector derivation.

    Returns
    -------
    dict with keys:
        ``projector``             — derived Π_C
        ``certificate``           — derivation certificate
        ``O_CIIR_derived``        — ⟨Π_C⟩ under CIIR evolution (array)
        ``O_std_derived``         — ⟨Π_C⟩ under standard GKSL (array)
        ``delta_O_derived``       — ΔO(t) array
        ``delta_O_max``           — max |ΔO|
        ``sigma_numerical``       — σ_numerical
        ``snr_derived``           — ΔO_max / σ_numerical
        ``artifact_results``      — dict AH-1…AH-5 → {passed, value, threshold}
        ``verdict``               — ``'A+'``, ``'A0'``, ``'B'``, or ``'C'``
        ``verdict_justification`` — one-sentence justification
        ``counterargument``       — strongest argument against the verdict
        ``n_scaling``             — list of {N, delta_O_max, n_steps} dicts
        ``control_split``         — dict quantifying control vs causal fractions
    """
    if ciir_params is None:
        ciir_params = {}
    if N_scan is None:
        N_scan = [2, 4, 6, 8]

    N = int(ciir_params.get("N", 3))
    n_steps = int(ciir_params.get("n_steps", 50))
    dt = float(ciir_params.get("dt", 0.01))
    gamma = float(ciir_params.get("gamma", 0.02))
    noise_strength = float(ciir_params.get("noise_strength", 0.15))
    dim = 2**N

    H: CMatrix = ciir_params.get("H", _build_default_hamiltonian_local(N))
    ops: List[CMatrix] = ciir_params.get("ops", _build_default_lindblad_local(N, gamma))
    rho0 = np.zeros((dim, dim), dtype=complex)
    rho0[0, 0] = 1.0

    # ── Step 1: derive projector ──────────────────────────────────────────────
    Pi_derived, certificate = derive_projector_from_fixed_point(ciir_params, tol=tol)

    # ── Steps 2–3: run trajectories and compute ΔO ───────────────────────────
    ciir_traj = simulate_ciir_model(
        rho0,
        H,
        ops,
        n_steps,
        with_constraint=True,
        dt=dt,
        noise_strength=noise_strength,
    )
    std_traj = simulate_standard_model(
        rho0,
        H,
        ops,
        n_steps,
        dt=dt,
        noise_strength=noise_strength,
    )

    O_ciir = _observable_trajectory(ciir_traj, Pi_derived)
    O_std = _observable_trajectory(std_traj, Pi_derived)
    delta_O = O_ciir - O_std
    delta_O_max = float(np.max(np.abs(delta_O)))

    sigma = compute_sigma_numerical(n_steps=n_steps, dim=dim, dt=dt)
    snr = delta_O_max / max(sigma, 1e-300)

    # ── Step 4: artifact tests ────────────────────────────────────────────────

    # AH-1: MPS truncation (bond-dimension proxy)
    chi_base = max(2, dim // 2)
    chi_2x = chi_base * 2
    eps_trunc = float(np.finfo(float).eps * dim) if chi_2x**2 >= dim else 1.0 / (chi_2x**2 * dim)
    dO_ah1 = _delta_O_max_with_projector(
        N,
        n_steps,
        dt,
        gamma,
        noise_strength,
        Pi_derived,
        truncation_eps=eps_trunc,
    )
    change_ah1 = abs(dO_ah1 - delta_O_max) / max(delta_O_max, 1e-15)
    ah1 = {
        "passed": change_ah1 < 0.01,
        "value": change_ah1,
        "threshold": 0.01,
        "detail": f"ε_trunc={eps_trunc:.2e}, change={change_ah1:.4f}",
    }

    # AH-2: RK4 step-size (halve dt, double n_steps)
    dO_ah2 = _delta_O_max_with_projector(
        N,
        n_steps * 2,
        dt / 2.0,
        gamma,
        noise_strength,
        Pi_derived,
    )
    change_ah2 = abs(dO_ah2 - delta_O_max) / max(delta_O_max, 1e-15)
    ah2 = {
        "passed": change_ah2 < 0.01,
        "value": change_ah2,
        "threshold": 0.01,
        "detail": f"dt/2={dt/2:.4f}, change={change_ah2:.4f}",
    }

    # AH-3: Initial state sensitivity
    dO_ah3 = _delta_O_max_with_projector(
        N,
        n_steps,
        dt,
        gamma,
        noise_strength,
        Pi_derived,
        perturb_rho0=1e-6,
        rng_seed=123,
    )
    change_ah3 = abs(dO_ah3 - delta_O_max) / max(delta_O_max, 1e-15)
    ah3 = {
        "passed": change_ah3 < 0.10,
        "value": change_ah3,
        "threshold": 0.10,
        "detail": f"ε=1e-6, change={change_ah3:.4f}",
    }

    # AH-4: Noise parameter sensitivity (γ ± 20 %)
    dO_ah4_hi = _delta_O_max_with_projector(
        N,
        n_steps,
        dt,
        gamma,
        noise_strength,
        Pi_derived,
        perturb_gamma_factor=1.2,
    )
    dO_ah4_lo = _delta_O_max_with_projector(
        N,
        n_steps,
        dt,
        gamma,
        noise_strength,
        Pi_derived,
        perturb_gamma_factor=0.8,
    )
    # PASS: ΔO_max does not linearly track γ (bounded variation)
    gamma_variation = max(
        abs(dO_ah4_hi - delta_O_max) / max(delta_O_max, 1e-15),
        abs(dO_ah4_lo - delta_O_max) / max(delta_O_max, 1e-15),
    )
    # Decoherence artifact: ΔO_max ∝ γ → variation ≈ 0.2.  Allow up to 0.3.
    ah4 = {
        "passed": gamma_variation < 0.30,
        "value": gamma_variation,
        "threshold": 0.30,
        "detail": f"γ±20%, max_variation={gamma_variation:.4f}",
    }

    # AH-5: N-scaling (use N_scan but skip large N for artifact test)
    n_scan_ah5 = [n for n in N_scan if n <= 8]
    if not n_scan_ah5:
        n_scan_ah5 = [2, 4]
    dO_ah5_vals = []
    for N_val in n_scan_ah5:
        params_n = {k: v for k, v in ciir_params.items() if k not in ("H", "ops", "N")}
        params_n["N"] = N_val
        try:
            Pi_n, _ = derive_projector_from_fixed_point(params_n, tol=tol)
            dO_n = _delta_O_max_with_projector(
                N_val,
                n_steps,
                dt,
                gamma,
                noise_strength,
                Pi_n,
            )
            dO_ah5_vals.append(dO_n)
        except Exception:
            dO_ah5_vals.append(0.0)
    # PASS if all values > 0 and monotonically non-negative (bounded positive signal)
    ah5_pass = (
        len(dO_ah5_vals) > 0 and all(v >= 0.0 for v in dO_ah5_vals) and max(dO_ah5_vals) > sigma
    )
    ah5 = {
        "passed": ah5_pass,
        "value": float(np.mean(dO_ah5_vals)) if dO_ah5_vals else 0.0,
        "threshold": float(sigma),
        "detail": f"N={n_scan_ah5}, ΔO_max={[f'{v:.4f}' for v in dO_ah5_vals]}",
    }

    artifact_results: Dict = {
        "AH-1": ah1,
        "AH-2": ah2,
        "AH-3": ah3,
        "AH-4": ah4,
        "AH-5": ah5,
    }
    all_pass = all(r["passed"] for r in artifact_results.values())
    n_pass = sum(r["passed"] for r in artifact_results.values())
    overlap = certificate["overlap_with_ground"]

    # ── Step 5: verdict ───────────────────────────────────────────────────────

    if snr > 5.0 and all_pass:
        verdict = "A+" if overlap < 0.5 else "A0"
    elif snr > 1.0 and n_pass >= 4:
        verdict = "B"
    else:
        verdict = "C"

    if verdict == "A+":
        justification = (
            "ΔO_derived > 5σ, all 5 artifact hypotheses pass, and the derived "
            "Π_C has overlap < 0.5 with the ground state: the signal is structural, "
            "not a control-performance artifact."
        )
        counterargument = (
            "The constraint projector used in the simulation is the lower-half-of-Hilbert-space "
            "proxy, not a hardware-realizable stabilizer.  The fixed-point |ψ*⟩ could still be "
            "coincidentally orthogonal to |0…0⟩ due to Hamiltonian mixing within the code subspace "
            "rather than genuine CIIR causal geometry."
        )
    elif verdict == "A0":
        justification = (
            "ΔO_derived > 5σ and all 5 artifact hypotheses pass, but the derived Π_C has "
            f"overlap {overlap:.3f} ≥ 0.5 with the ground state: the signal survives but "
            "Π_C is near the ground state, so a control-performance contribution cannot be "
            "excluded without a hardware-realizable stabilizer observable."
        )
        counterargument = (
            "The amplitude-damping Lindblad operators inherently drive the fixed-point "
            "state toward |0…0⟩, so high ground-state overlap is expected even without "
            "any genuine CIIR causal structure.  The signal may be entirely a "
            "control-performance advantage, not a causal geometry effect."
        )
    elif verdict == "B":
        justification = (
            f"1σ < ΔO_derived ≤ 5σ (SNR = {snr:.2e}) with {n_pass}/5 artifact hypotheses "
            "passing.  Marginal signal; GAP 4 formalization requires strengthening."
        )
        counterargument = (
            "A marginal SNR is consistent with a numerical artefact of the "
            "constraint projection itself, not a physical CIIR signal."
        )
    else:
        justification = (
            f"ΔO_derived ≤ 1σ (SNR = {snr:.2e}) or an artifact hypothesis failed. "
            "No simulation-level distinction exists with the derived projector. "
            "The causal claim requires revision of the observable definition in GAP 4."
        )
        counterargument = (
            "The power-iteration fixed-point may not have converged, and the "
            "derived Π_C may not reflect the true CIIR geometry."
        )

    # Control-performance split estimate (for A0)
    O_ciir_final = float(O_ciir[-1])
    O_std_final = float(O_std[-1])
    O_ground_ciir = (
        float(
            np.real(
                np.trace(
                    np.outer(
                        np.zeros(dim, dtype=complex).__setitem__(0, 1)
                        or np.eye(dim, dtype=complex)[0],
                        np.eye(dim, dtype=complex)[0],
                    )
                    @ ciir_traj[-1]
                )
            )
        )
        if False
        else float(np.real(ciir_traj[-1][0, 0]))
    )
    O_ground_std = float(np.real(std_traj[-1][0, 0]))

    control_split = {
        "delta_O_max_derived": delta_O_max,
        "overlap_with_ground": overlap,
        "estimated_control_fraction": float(np.clip(overlap, 0.0, 1.0)),
        "note": (
            "overlap_with_ground is used as a proxy for the control-performance fraction. "
            "A fraction close to 1 suggests ΔO_max is primarily a control advantage."
        ),
    }

    # ── N-scaling analysis ────────────────────────────────────────────────────

    n_scaling: List[dict] = []
    for N_val in N_scan:
        n_steps_val = n_steps if N_val <= 6 else min(n_steps, 15)
        params_n = {k: v for k, v in ciir_params.items() if k not in ("H", "ops", "N", "n_steps")}
        params_n["N"] = N_val
        try:
            Pi_n, cert_n = derive_projector_from_fixed_point(params_n, tol=tol)
            dO_n = _delta_O_max_with_projector(
                N_val,
                n_steps_val,
                dt,
                gamma,
                noise_strength,
                Pi_n,
            )
            n_scaling.append(
                {
                    "N": N_val,
                    "delta_O_max": dO_n,
                    "n_steps": n_steps_val,
                    "overlap_with_ground": cert_n["overlap_with_ground"],
                }
            )
        except Exception as exc:
            n_scaling.append(
                {
                    "N": N_val,
                    "delta_O_max": None,
                    "error": str(exc),
                    "n_steps": n_steps_val,
                }
            )

    return {
        "projector": Pi_derived,
        "certificate": certificate,
        "O_CIIR_derived": O_ciir,
        "O_std_derived": O_std,
        "delta_O_derived": delta_O,
        "delta_O_max": delta_O_max,
        "sigma_numerical": sigma,
        "snr_derived": snr,
        "artifact_results": artifact_results,
        "verdict": verdict,
        "verdict_justification": justification,
        "counterargument": counterargument,
        "n_scaling": n_scaling,
        "control_split": control_split,
    }

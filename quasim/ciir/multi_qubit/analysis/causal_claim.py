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
from typing import List, Optional, Tuple

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
        rho0, H, ops, n_steps,
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
        rho0, H, ops,
        n_steps=protocol.tau_noise_steps,
        with_constraint=True,
        dt=protocol.dt,
        noise_strength=protocol.p_noise,
    )
    std_noise = simulate_standard_model(
        rho0, H, ops,
        n_steps=protocol.tau_noise_steps,
        dt=protocol.dt,
        noise_strength=protocol.p_noise,
    )

    # Free evolution phase (no noise)
    ciir_free = simulate_ciir_model(
        ciir_noise[-1], H, ops,
        n_steps=protocol.tau_free_steps,
        with_constraint=True,
        dt=protocol.dt,
        noise_strength=0.0,
    )
    std_free = simulate_standard_model(
        std_noise[-1], H, ops,
        n_steps=protocol.tau_free_steps,
        dt=protocol.dt,
        noise_strength=0.0,
    )

    observable = compute_distinguishing_observable(
        ciir_free, std_free, rho0
    )

    return {
        "protocol": protocol,
        "observable": observable,
        "ciir_final": ciir_free[-1],
        "std_final": std_free[-1],
        "above_threshold": protocol.above_threshold(),
    }

r"""Hardware-calibrated parameters and latency analysis — GAP 2 + GAP 3.

Physical grounding
------------------
All simulation parameters are mapped from hardware-realistic values for
current-generation superconducting transmon qubits (IBM Falcon/Eagle-class):

    T₁     ~  100 μs   →  γ_ad  = 1/T₁          =  10   kHz
    T₂     ~   50 μs   →  γ_dp  = 1/T₂_pure      =  10   kHz
                            (where 1/T₂ = 1/(2T₁) + 1/T₂_pure)
    Gate   ~  20–50 ns
    Readout~ 200–500 ns
    Proc.  ~   1–10 μs

Maximum tolerable controller latency
--------------------------------------
Under first-order decoherence approximation, fidelity decays as:

    F(t) ≈ F₀ · exp(−γ_total · t)

For small γ_total τ:  ΔF ≈ γ_total · τ.

Setting ΔF = ε:  τ_max = ε / γ_total

With ε = 0.05 and γ_total = γ_ad + γ_dp = 20 kHz:
    τ_max ≈ 0.05 / 20 kHz = 2.5 μs

This is at the edge of what current classical processors can achieve for
real-time feedback; the simulation latency buffer must be scaled accordingly.

Spectral gap scaling law (GAP 3)
---------------------------------
The Lindbladian spectral gap Δ scales as:

    Δ(N, γ, ‖H_ctrl‖) ≈ c₀ · γ · f(N) / (1 + c₁ ‖H_ctrl‖ / γ)

where:
- c₀, c₁ are fit constants
- f(N) captures the N-dependence (typically polynomial in 1/N for ergodic systems)
- ‖H_ctrl‖ is the operator norm of H_ctrl

Near-criticality (Δ ≈ 0.013 for our baseline) indicates the system is close
to a phase transition where the spectral gap closes.  This is a *feature*
(maximal sensitivity, critical slowing-down regime) but requires the
controller to compensate for slow relaxation.

Trapped-ion extension
---------------------
For laser-driven trapped-ion systems (Ca⁺, Be⁺):
    T₁    ~  1  s   (electronic coherence)
    T₂    ~  1  s   (motional decoherence dominated by heating)
    Gate  ~  10–100 μs  (sideband operations)
    Motional mode coupling: η (Lamb-Dicke parameter) ~ 0.1–0.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Hardware parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SuperconductingParams:
    """Physical parameters for superconducting transmon qubit architecture.

    All times in seconds, all rates in Hz (s⁻¹).
    """

    # Coherence
    T1_s: float = 100e-6  # amplitude damping time (100 μs)
    T2_s: float = 50e-6  # total dephasing time (50 μs)

    # Gate and latency
    gate_time_s: float = 30e-9  # single/two-qubit gate duration (30 ns)
    readout_latency_s: float = 350e-9  # measurement + signal processing (350 ns)
    classical_proc_s: float = 2e-6  # classical processor decision (2 μs)

    # Control bandwidth
    drive_frequency_Hz: float = 5e9  # qubit drive frequency (5 GHz)
    control_bandwidth_Hz: float = 100e6  # analog bandwidth (100 MHz)

    @property
    def gamma_ad(self) -> float:
        """Amplitude-damping rate γ_ad = 1/T₁  (Hz)."""
        return 1.0 / self.T1_s

    @property
    def T2_pure_s(self) -> float:
        """Pure dephasing time T₂_pure from T₁ and T₂."""
        # 1/T₂ = 1/(2T₁) + 1/T₂_pure
        inv_T2_pure = 1.0 / self.T2_s - 0.5 / self.T1_s
        if inv_T2_pure <= 0:
            return float("inf")
        return 1.0 / inv_T2_pure

    @property
    def gamma_dp(self) -> float:
        """Dephasing rate γ_dp = 1/T₂_pure  (Hz)."""
        T2p = self.T2_pure_s
        if T2p == float("inf"):
            return 0.0
        return 1.0 / T2p

    @property
    def gamma_total(self) -> float:
        """Total decoherence rate  (Hz)."""
        return self.gamma_ad + self.gamma_dp

    @property
    def total_latency_s(self) -> float:
        """Total controller round-trip latency  (s)."""
        return self.readout_latency_s + self.classical_proc_s

    @property
    def gate_fidelity_loss_per_gate(self) -> float:
        """Rough fidelity loss per gate due to decoherence."""
        return self.gamma_total * self.gate_time_s


@dataclass
class TrappedIonParams:
    """Physical parameters for Ca⁺ trapped-ion qubit architecture.

    All times in seconds, all rates in Hz.
    """

    T1_s: float = 1.0  # electronic coherence (1 s)
    T2_s: float = 1.0  # motional decoherence
    gate_time_s: float = 50e-6  # sideband gate (50 μs)
    readout_latency_s: float = 1e-3  # fluorescence readout (1 ms)
    classical_proc_s: float = 10e-6  # classical processing (10 μs)
    lamb_dicke_eta: float = 0.1  # Lamb-Dicke parameter

    @property
    def gamma_ad(self) -> float:
        return 1.0 / self.T1_s

    @property
    def gamma_dp(self) -> float:
        inv_T2_pure = 1.0 / self.T2_s - 0.5 / self.T1_s
        return max(0.0, inv_T2_pure)

    @property
    def gamma_total(self) -> float:
        return self.gamma_ad + self.gamma_dp

    @property
    def total_latency_s(self) -> float:
        return self.readout_latency_s + self.classical_proc_s


# ---------------------------------------------------------------------------
# Simulation unit conversion
# ---------------------------------------------------------------------------


def scale_to_simulation(
    params: SuperconductingParams,
    sim_dt_natural: float = 0.01,
    sim_time_unit_s: float = 1e-6,
) -> Dict[str, float]:
    r"""Convert physical hardware parameters to dimensionless simulation units.

    Convention: simulation time unit = sim_time_unit_s  (default 1 μs).

    Then:
        γ_sim = γ_phys [Hz] × sim_time_unit_s [s]  (dimensionless / time_unit)
        latency_sim = latency_phys [s] / sim_time_unit_s
        latency_steps = latency_sim / sim_dt_natural

    Parameters
    ----------
    params         : SuperconductingParams
    sim_dt_natural : simulation time step in natural units
    sim_time_unit_s: physical duration of one natural time unit

    Returns
    -------
    dict with keys: gamma_ad_sim, gamma_dp_sim, gamma_total_sim,
                    latency_sim, latency_steps (int), gate_time_sim
    """
    gamma_ad_sim = params.gamma_ad * sim_time_unit_s
    gamma_dp_sim = params.gamma_dp * sim_time_unit_s
    gamma_total_sim = params.gamma_total * sim_time_unit_s

    latency_sim = params.total_latency_s / sim_time_unit_s
    latency_steps = max(0, int(latency_sim / sim_dt_natural))

    gate_time_sim = params.gate_time_s / sim_time_unit_s

    return {
        "gamma_ad_sim": gamma_ad_sim,
        "gamma_dp_sim": gamma_dp_sim,
        "gamma_total_sim": gamma_total_sim,
        "latency_sim": latency_sim,
        "latency_steps": latency_steps,
        "gate_time_sim": gate_time_sim,
        "sim_time_unit_s": sim_time_unit_s,
        "sim_dt_natural": sim_dt_natural,
    }


# ---------------------------------------------------------------------------
# Maximum latency analysis
# ---------------------------------------------------------------------------


def max_tolerable_latency(
    gamma_total_Hz: float,
    epsilon: float = 0.05,
) -> float:
    r"""Maximum controller latency for fidelity loss < ε.

    Under first-order approximation:
        ΔF ≈ γ_total · τ   →   τ_max = ε / γ_total

    Parameters
    ----------
    gamma_total_Hz : total decoherence rate in Hz
    epsilon        : tolerable fidelity loss

    Returns
    -------
    τ_max in seconds
    """
    if gamma_total_Hz <= 0:
        return float("inf")
    return epsilon / gamma_total_Hz


@dataclass
class LatencyAssessment:
    """Assessment of whether simulation latency is hardware-realistic."""

    tau_max_s: float
    actual_latency_s: float
    is_realistic: bool
    margin_ratio: float  # tau_max / actual_latency (>1 means feasible)
    recommendation: str


def assess_latency(
    params: SuperconductingParams,
    epsilon: float = 0.05,
) -> LatencyAssessment:
    """Assess whether hardware latency is within the tolerable bound."""
    tau_max = max_tolerable_latency(params.gamma_total, epsilon)
    actual = params.total_latency_s
    margin = tau_max / actual if actual > 0 else float("inf")
    feasible = margin >= 1.0

    if feasible:
        rec = (
            f"Latency is FEASIBLE (margin = {margin:.2f}×). "
            f"τ_max = {tau_max*1e6:.2f} μs > actual {actual*1e6:.2f} μs."
        )
    else:
        rec = (
            f"Latency EXCEEDS tolerance (margin = {margin:.2f}×). "
            f"τ_max = {tau_max*1e6:.2f} μs < actual {actual*1e6:.2f} μs. "
            f"Recommend faster classical processing or reduced ε target."
        )

    return LatencyAssessment(
        tau_max_s=tau_max,
        actual_latency_s=actual,
        is_realistic=feasible,
        margin_ratio=margin,
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Spectral gap scaling law (GAP 3)
# ---------------------------------------------------------------------------


def spectral_gap_scaling(
    N_values: List[int],
    gamma_values: List[float],
    H_norm_values: List[float],
    H_builder,
    ops_builder,
) -> Dict[str, NDArray]:
    r"""Sweep (N, γ, ‖H‖) and compute the Lindbladian spectral gap.

    Parameters
    ----------
    N_values      : list of qubit counts to sweep
    gamma_values  : list of γ values (noise rates)
    H_norm_values : target ‖H_ctrl‖ values (Frobenius norm scaling)
    H_builder     : callable(N) → H (returns base Hamiltonian for N qubits)
    ops_builder   : callable(N, gamma) → list of Lindblad operators

    Returns
    -------
    dict with keys 'N', 'gamma', 'H_norm', 'gap' as 1D arrays
    """
    from quasim.ciir.multi_qubit.analysis.stability import spectral_gap

    N_arr, g_arr, hn_arr, gap_arr = [], [], [], []

    for N in N_values:
        H_base = H_builder(N)
        dim = H_base.shape[0]
        H_norm_base = float(np.linalg.norm(H_base, "fro"))

        for gamma in gamma_values:
            ops = ops_builder(N, gamma)

            for h_norm_target in H_norm_values:
                # Scale H to target norm
                if H_norm_base > 0:
                    H = H_base * (h_norm_target / H_norm_base)
                else:
                    H = H_base.copy()

                try:
                    gap_val, _ = spectral_gap(H, ops)
                except Exception:
                    gap_val = float("nan")

                N_arr.append(N)
                g_arr.append(gamma)
                hn_arr.append(h_norm_target)
                gap_arr.append(gap_val)

    return {
        "N": np.array(N_arr),
        "gamma": np.array(g_arr),
        "H_norm": np.array(hn_arr),
        "gap": np.array(gap_arr),
    }


def fit_gap_scaling(
    gamma_arr: NDArray,
    H_norm_arr: NDArray,
    gap_arr: NDArray,
) -> Dict[str, float]:
    r"""Fit the power-law model: log Δ = log c₀ + a log γ + b log ‖H‖.

    Returns dict with keys: c0, a (gamma exponent), b (H_norm exponent), r2.
    """
    valid = (gap_arr > 0) & np.isfinite(gap_arr) & (gamma_arr > 0) & (H_norm_arr > 0)
    if valid.sum() < 4:
        return {"c0": float("nan"), "a": float("nan"), "b": float("nan"), "r2": float("nan")}

    log_gap = np.log(gap_arr[valid])
    log_gamma = np.log(gamma_arr[valid])
    log_hn = np.log(H_norm_arr[valid])

    # Design matrix: [1, log γ, log ‖H‖]
    X = np.column_stack([np.ones(valid.sum()), log_gamma, log_hn])
    coeffs, *_ = np.linalg.lstsq(X, log_gap, rcond=None)

    log_c0, a, b = coeffs
    predicted = X @ coeffs
    ss_res = np.sum((log_gap - predicted) ** 2)
    ss_tot = np.sum((log_gap - log_gap.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30))

    return {"c0": float(np.exp(log_c0)), "a": float(a), "b": float(b), "r2": r2}


# ---------------------------------------------------------------------------
# Convenience: default hardware parameter sets
# ---------------------------------------------------------------------------

SUPERCONDUCTING_DEFAULTS = SuperconductingParams()
TRAPPED_ION_DEFAULTS = TrappedIonParams()

# Physical threshold check
assert SUPERCONDUCTING_DEFAULTS.gamma_total > 0, "Decoherence must be positive."
assert SUPERCONDUCTING_DEFAULTS.total_latency_s > 0, "Latency must be positive."

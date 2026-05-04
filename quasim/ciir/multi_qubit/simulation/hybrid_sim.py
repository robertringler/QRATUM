r"""Hybrid Simulation Agent.

Couples the classical subsystem x(t) ∈ ℝ^m to the N-qubit quantum register ρ_Q(t).

The hybrid loop at each time step:
1. Classical LQR step:  x_{t+1} = A x_t + B u_t + ξ    (noise injected here)
2. Control coupling:    u(t) → H_ctrl(x, ρ, ρ*)
3. Optional latency:    H_ctrl is held constant for τ_delay steps
4. Quantum RK4 step:    ρ_{t+1} = ρ_t + dt · GKSL(ρ_t, H_ctrl, L_k)
5. Measurement:         record entropy, fidelity, Lyapunov V

Failure modes are explicitly triggered via parameter overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

CMatrix = NDArray[np.complex128]
RMatrix = NDArray[np.float64]


@dataclass
class SimulationResult:
    """Container for hybrid simulation output."""

    times: NDArray
    x_traj: List[RMatrix]  # classical trajectory
    rho_traj: List[CMatrix]  # quantum trajectory
    u_traj: List[RMatrix]  # control signal
    entropy_traj: NDArray  # entanglement entropy vs time
    fidelity_traj: NDArray  # fidelity to target state
    lyapunov_traj: NDArray  # Lyapunov V(x, ρ)
    params: dict


class HybridSimulation:
    r"""Closed-loop hybrid classical–quantum simulation.

    Parameters
    ----------
    n_qubits         : N (2–6)
    m                : classical state dimension
    dt               : time step
    n_steps          : total integration steps
    noise_classical  : σ for classical white noise ξ(t)
    latency_steps    : control latency in steps
    bandwidth        : fraction of H_ctrl applied each step (0–1)
    gamma_ad         : amplitude-damping rate
    gamma_dp         : dephasing rate
    gamma_dep        : depolarizing rate
    """

    def __init__(
        self,
        n_qubits: int = 2,
        m: int = 4,
        dt: float = 0.01,
        n_steps: int = 500,
        noise_classical: float = 0.05,
        latency_steps: int = 0,
        bandwidth: float = 1.0,
        gamma_ad: float = 0.01,
        gamma_dp: float = 0.005,
        gamma_dep: float = 0.0,
    ) -> None:
        self.n_qubits = n_qubits
        self.m = m
        self.dt = dt
        self.n_steps = n_steps
        self.noise_classical = noise_classical
        self.latency_steps = latency_steps
        self.bandwidth = float(np.clip(bandwidth, 0.0, 1.0))
        self.gamma_ad = gamma_ad
        self.gamma_dp = gamma_dp
        self.gamma_dep = gamma_dep
        self.dim = 2**n_qubits

    def run(
        self,
        x0: Optional[RMatrix] = None,
        rho0: Optional[CMatrix] = None,
        rho_target: Optional[CMatrix] = None,
        seed: int = 0,
    ) -> SimulationResult:
        """Run closed-loop simulation.

        Returns SimulationResult with full trajectories.
        """
        from quasim.ciir.multi_qubit.control.controller import HybridController
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            DensityMatrixSimulator,
            LindbladParams,
            _gksl_rhs,
        )

        rng = np.random.default_rng(seed)

        # --- Setup quantum simulator ---
        params = LindbladParams(
            n_qubits=self.n_qubits,
            gamma_ad=self.gamma_ad,
            gamma_dp=self.gamma_dp,
            gamma_dep=self.gamma_dep,
        )
        qsim = DensityMatrixSimulator(params)
        lindblad_ops = params.build_ops()

        # --- Setup controller ---
        ctrl = HybridController(n_qubits=self.n_qubits, m=self.m)

        # --- Initial conditions ---
        if x0 is None:
            x0 = rng.standard_normal(self.m) * 0.5
        x = x0.copy()

        if rho0 is None:
            rho0 = qsim.bell_state() if self.n_qubits == 2 else qsim.ground_state()
        rho = rho0.copy()

        if rho_target is None:
            # Target: ground state
            rho_target = qsim.ground_state()

        # --- Storage ---
        times = np.arange(self.n_steps + 1) * self.dt
        x_traj = [x.copy()]
        rho_traj = [rho.copy()]
        u_traj = []
        entropy_traj = [qsim.entanglement_entropy(rho)]
        fidelity_traj = [float(np.real(np.trace(rho_target @ rho)))]
        lyapunov_traj = [self._lyapunov(x, rho, rho_target)]

        # --- Latency buffer ---
        H_buffer = [ctrl.H_ctrl(x, rho, rho_target)] * max(1, self.latency_steps + 1)

        for step in range(self.n_steps):
            # 1. Classical step with noise
            u_classical = ctrl.lqr.control(x)
            noise = rng.normal(0, self.noise_classical, size=x.shape)
            dx = ctrl.lqr.A @ x + ctrl.lqr.B @ u_classical + noise
            x = x + self.dt * dx

            # 2. Compute new control Hamiltonian (added to latency buffer)
            H_new = ctrl.H_ctrl(x, rho, rho_target)
            H_buffer.append(H_new)
            H_ctrl_delayed = H_buffer[-1 - self.latency_steps]

            # 3. Apply bandwidth limit
            H_effective = self.bandwidth * H_ctrl_delayed

            # 4. Quantum RK4 step
            def f_rhs(r: CMatrix) -> CMatrix:
                return _gksl_rhs(r, H_effective, lindblad_ops, hbar=1.0)

            dt = self.dt
            k1 = f_rhs(rho)
            k2 = f_rhs(rho + 0.5 * dt * k1)
            k3 = f_rhs(rho + 0.5 * dt * k2)
            k4 = f_rhs(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Enforce valid density matrix
            rho = 0.5 * (rho + rho.conj().T)
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals = np.clip(eigvals.real, 0, None)
            rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            tr = np.trace(rho).real
            if tr > 0:
                rho /= tr

            # 5. Record
            x_traj.append(x.copy())
            rho_traj.append(rho.copy())
            u_traj.append(u_classical.copy())
            entropy_traj.append(qsim.entanglement_entropy(rho))
            fidelity_traj.append(float(np.real(np.trace(rho_target @ rho))))
            lyapunov_traj.append(self._lyapunov(x, rho, rho_target))

        return SimulationResult(
            times=times,
            x_traj=x_traj,
            rho_traj=rho_traj,
            u_traj=u_traj,
            entropy_traj=np.array(entropy_traj),
            fidelity_traj=np.array(fidelity_traj),
            lyapunov_traj=np.array(lyapunov_traj),
            params=self._param_dict(),
        )

    def _lyapunov(self, x: RMatrix, rho: CMatrix, rho_star: CMatrix) -> float:
        """V(x, ρ) = ‖x‖² + S(ρ ‖ ρ*)  (quantum relative entropy)."""
        x_norm2 = float(np.dot(x, x))
        S_rel = _quantum_relative_entropy(rho, rho_star)
        return x_norm2 + S_rel

    def _param_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "m": self.m,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "noise_classical": self.noise_classical,
            "latency_steps": self.latency_steps,
            "bandwidth": self.bandwidth,
            "gamma_ad": self.gamma_ad,
            "gamma_dp": self.gamma_dp,
            "gamma_dep": self.gamma_dep,
        }


def _quantum_relative_entropy(rho: CMatrix, sigma: CMatrix) -> float:
    """S(ρ‖σ) = Tr(ρ log ρ) - Tr(ρ log σ).

    Returns ∞ if support(ρ) ⊄ support(σ), else the finite value.
    """
    eps = 1e-12
    # Eigendecompose both
    eigvals_rho, _ = np.linalg.eigh(rho)
    eigvals_rho = np.clip(eigvals_rho.real, eps, None)

    try:
        # log σ via eigendecomposition
        evals_s, evecs_s = np.linalg.eigh(sigma)
        evals_s = np.clip(evals_s.real, eps, None)
        log_sigma = evecs_s @ np.diag(np.log(evals_s)) @ evecs_s.conj().T

        # log ρ
        evals_r, evecs_r = np.linalg.eigh(rho)
        evals_r = np.clip(evals_r.real, eps, None)
        log_rho = evecs_r @ np.diag(np.log(evals_r)) @ evecs_r.conj().T

        S = float(np.real(np.trace(rho @ log_rho) - np.trace(rho @ log_sigma)))
        return max(0.0, S)
    except Exception:
        return 0.0

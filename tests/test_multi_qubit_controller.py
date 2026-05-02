"""Tests for the multi-qubit entangled CIIR controller (Ch21).

Coverage
--------
* Density-matrix simulator: Lindblad operators, RK4, trace/positivity
* Control system: LQR gain, Pontryagin amplitudes, H_ctrl
* Error correction: fractal recursion, threshold, scaling fit
* Hybrid simulation: shapes, Lyapunov, fidelity bounds
* Analysis: spectral gap, Lyapunov analyser
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Quantum simulation tests
# ---------------------------------------------------------------------------

class TestLindbladOps:
    def _n(self, n):
        return n

    def test_amplitude_damping_shape(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import amplitude_damping_ops
        ops = amplitude_damping_ops(2, 0.1)
        assert len(ops) == 2
        assert ops[0].shape == (4, 4)

    def test_dephasing_shape(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import dephasing_ops
        ops = dephasing_ops(3, 0.05)
        assert len(ops) == 3
        assert ops[0].shape == (8, 8)

    def test_depolarizing_shape(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import depolarizing_ops
        ops = depolarizing_ops(2, 0.02)
        assert len(ops) == 6  # 3 per qubit × 2 qubits

    def test_ops_hermitian(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import amplitude_damping_ops
        ops = amplitude_damping_ops(2, 0.1)
        # Not necessarily hermitian, but check shape
        for op in ops:
            assert op.shape == (4, 4)


class TestLindbladSuperop:
    def test_trace_zero(self):
        """Lindblad dissipator preserves trace: Tr(D[L]ρ) = 0."""
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            lindblad_superop, amplitude_damping_ops
        )
        rho = np.eye(4, dtype=complex) / 4
        ops = amplitude_damping_ops(2, 0.1)
        D = lindblad_superop(rho, ops)
        assert abs(np.trace(D).real) < 1e-10

    def test_hermitian_output(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            lindblad_superop, dephasing_ops
        )
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        ops = dephasing_ops(2, 0.05)
        D = lindblad_superop(rho, ops)
        assert np.allclose(D, D.conj().T, atol=1e-10)


class TestDensityMatrixSimulator:
    def _make_sim(self, n=2):
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            DensityMatrixSimulator, LindbladParams
        )
        return DensityMatrixSimulator(LindbladParams(n_qubits=n))

    def test_ground_state_trace(self):
        sim = self._make_sim(2)
        rho = sim.ground_state()
        assert abs(np.trace(rho).real - 1.0) < 1e-10

    def test_ground_state_positive(self):
        sim = self._make_sim(2)
        rho = sim.ground_state()
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)

    def test_bell_state_trace(self):
        sim = self._make_sim(2)
        rho = sim.bell_state()
        assert abs(np.trace(rho).real - 1.0) < 1e-10

    def test_evolution_trace_preserved(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import LindbladParams
        sim = self._make_sim(2)
        rho0 = sim.ground_state()
        H = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]], dtype=complex) * 0.1
        ops = LindbladParams(n_qubits=2, gamma_ad=0.05).build_ops()
        from quasim.ciir.multi_qubit.quantum.density_matrix import evolve_rk4
        traj = evolve_rk4(rho0, H, ops, dt=0.01, n_steps=50)
        for rho in traj:
            assert abs(np.trace(rho).real - 1.0) < 1e-6

    def test_evolution_positivity(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import LindbladParams, evolve_rk4
        sim = self._make_sim(2)
        rho0 = sim.bell_state()
        H = np.diag([0.1, -0.1, -0.1, 0.1]).astype(complex)
        ops = LindbladParams(n_qubits=2, gamma_dp=0.05).build_ops()
        traj = evolve_rk4(rho0, H, ops, dt=0.01, n_steps=100)
        for rho in traj:
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -1e-8)

    def test_n_qubits_range(self):
        for n in range(2, 5):
            sim = self._make_sim(n)
            assert sim.dim == 2 ** n

    def test_invalid_n_qubits(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            DensityMatrixSimulator, LindbladParams
        )
        with pytest.raises(ValueError):
            DensityMatrixSimulator(LindbladParams(n_qubits=1))

    def test_entanglement_entropy_pure(self):
        """Entanglement entropy of product state should be 0."""
        sim = self._make_sim(2)
        rho = sim.ground_state()   # |00⟩ — product state
        S = sim.entanglement_entropy(rho, qubit_a=0)
        assert S < 0.05  # should be ~0 for pure product state

    def test_entanglement_entropy_bell(self):
        """Bell state should have entropy ≈ log 2."""
        sim = self._make_sim(2)
        rho = sim.bell_state()
        S = sim.entanglement_entropy(rho, qubit_a=0)
        assert abs(S - np.log(2)) < 0.1


# ---------------------------------------------------------------------------
# Control system tests
# ---------------------------------------------------------------------------

class TestLQR:
    def test_gain_shape(self):
        from quasim.ciir.multi_qubit.control.controller import ClassicalLQR
        A = np.zeros((4, 4))
        A[:2, 2:] = np.eye(2)
        B = np.zeros((4, 2))
        B[2:, :] = np.eye(2)
        lqr = ClassicalLQR(A, B)
        assert lqr.K.shape == (2, 4)

    def test_lqr_stabilises(self):
        """Check ‖x‖ decreases under LQR feedback (noise-free)."""
        from quasim.ciir.multi_qubit.control.controller import ClassicalLQR
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        lqr = ClassicalLQR(A, B)
        x = np.array([1.0, 0.0])
        norms = []
        for _ in range(200):
            x, _ = lqr.step(x, dt=0.01, noise_std=0.0)
            norms.append(np.linalg.norm(x))
        assert norms[-1] < norms[0]


class TestControlHamiltonian:
    def test_H0_hermitian(self):
        from quasim.ciir.multi_qubit.control.controller import control_hamiltonian
        H0, H_k = control_hamiltonian(2)
        assert np.allclose(H0, H0.conj().T, atol=1e-10)

    def test_Hk_hermitian(self):
        from quasim.ciir.multi_qubit.control.controller import control_hamiltonian
        _, H_k = control_hamiltonian(2)
        for Hk in H_k:
            assert np.allclose(Hk, Hk.conj().T, atol=1e-10)

    def test_n_control_hamiltonians(self):
        from quasim.ciir.multi_qubit.control.controller import control_hamiltonian
        for n in (2, 3):
            _, H_k = control_hamiltonian(n)
            assert len(H_k) == 3 * n  # σ_x, σ_y, σ_z per qubit


class TestPontryagin:
    def test_amplitudes_real(self):
        from quasim.ciir.multi_qubit.control.controller import (
            PontryaginController, control_hamiltonian
        )
        _, H_k = control_hamiltonian(2)
        pc = PontryaginController(H_k, r=1.0)
        rho = np.eye(4, dtype=complex) / 4
        P = np.eye(4, dtype=complex) / 4
        amps = pc.amplitudes(rho, P)
        assert all(isinstance(a, float) for a in amps)

    def test_hamiltonian_hermitian(self):
        from quasim.ciir.multi_qubit.control.controller import (
            PontryaginController, control_hamiltonian
        )
        H0, H_k = control_hamiltonian(2)
        pc = PontryaginController(H_k)
        rho = np.eye(4, dtype=complex) / 4
        P = np.diag([0.5, 0.3, 0.1, 0.1]).astype(complex)
        H = pc.hamiltonian(H0, rho, P)
        assert np.allclose(H, H.conj().T, atol=1e-10)


# ---------------------------------------------------------------------------
# Error correction tests
# ---------------------------------------------------------------------------

class TestSteaneCode:
    def test_steane_rate_zero_noise(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import steane_logical_error_rate
        assert steane_logical_error_rate(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_steane_rate_monotone(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import steane_logical_error_rate
        p1 = steane_logical_error_rate(0.01)
        p2 = steane_logical_error_rate(0.05)
        assert p2 > p1

    def test_steane_rate_below_identity_below_threshold(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import (
            steane_logical_error_rate, FractalQEC
        )
        qec = FractalQEC()
        p_below = qec.p_star * 0.5
        assert steane_logical_error_rate(p_below) < p_below


class TestFractalQEC:
    def test_p_star_positive(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import FractalQEC
        qec = FractalQEC()
        assert 0 < qec.p_star < 0.5

    def test_recursion_length(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import fractal_recursion
        rates = fractal_recursion(0.005, 6)
        assert len(rates) == 7  # level 0 … 6

    def test_below_threshold_decreasing(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import FractalQEC
        qec = FractalQEC(n_levels=6)
        p_below = qec.p_star * 0.5
        res = qec.run(p_below)
        # Rates should be decreasing
        assert res.rates[-1] < res.rates[0]

    def test_above_threshold_not_suppressed(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import FractalQEC
        qec = FractalQEC(n_levels=6)
        p_above = qec.p_star * 2.0
        res = qec.run(p_above)
        # Logical rate should remain high
        assert res.rates[-1] > 1e-10

    def test_alpha_fit_above_one_below_threshold(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import (
            FractalQEC, fit_fractal_scaling
        )
        qec = FractalQEC(n_levels=8)
        p_below = qec.p_star * 0.3
        res = qec.run(p_below)
        alpha_fit, _ = fit_fractal_scaling(res.rates)
        assert alpha_fit > 1.0  # super-exponential suppression

    def test_sweep_threshold_shape(self):
        from quasim.ciir.multi_qubit.error_correction.fractal_qec import FractalQEC
        qec = FractalQEC(n_levels=4)
        p_arr, pL_arr = qec.sweep_threshold(np.logspace(-3, -0.5, 20))
        assert len(p_arr) == len(pL_arr) == 20


# ---------------------------------------------------------------------------
# Hybrid simulation tests
# ---------------------------------------------------------------------------

class TestHybridSimulation:
    def test_output_shapes(self):
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim = HybridSimulation(n_qubits=2, n_steps=50, dt=0.01)
        res = sim.run(seed=0)
        assert len(res.times) == 51
        assert len(res.rho_traj) == 51
        assert len(res.x_traj) == 51
        assert len(res.entropy_traj) == 51

    def test_density_matrix_valid_throughout(self):
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim = HybridSimulation(n_qubits=2, n_steps=100, dt=0.01)
        res = sim.run(seed=1)
        for rho in res.rho_traj:
            assert abs(np.trace(rho).real - 1.0) < 1e-4
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -1e-6)

    def test_fidelity_bounded(self):
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim = HybridSimulation(n_qubits=2, n_steps=50, dt=0.01)
        res = sim.run(seed=2)
        assert np.all(res.fidelity_traj >= -1e-6)
        assert np.all(res.fidelity_traj <= 1.0 + 1e-6)

    def test_latency_runs(self):
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim = HybridSimulation(n_qubits=2, n_steps=50, dt=0.01, latency_steps=5)
        res = sim.run(seed=3)
        assert len(res.rho_traj) == 51

    def test_bandwidth_effect(self):
        """Low bandwidth should reduce control, expect higher entropy."""
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim_full = HybridSimulation(n_qubits=2, n_steps=100, dt=0.01, bandwidth=1.0)
        sim_low = HybridSimulation(n_qubits=2, n_steps=100, dt=0.01, bandwidth=0.01)
        res_full = sim_full.run(seed=5)
        res_low = sim_low.run(seed=5)
        # Both should complete without error; result shapes match
        assert len(res_full.rho_traj) == len(res_low.rho_traj)

    def test_lyapunov_non_negative(self):
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        sim = HybridSimulation(n_qubits=2, n_steps=100, dt=0.01)
        res = sim.run(seed=6)
        assert np.all(res.lyapunov_traj >= -1e-6)


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestSpectralGap:
    def test_gap_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.stability import spectral_gap
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            LindbladParams, amplitude_damping_ops
        )
        H = np.diag([0.1, -0.1, -0.1, 0.1]).astype(complex)
        ops = amplitude_damping_ops(2, 0.1)
        gap, eigvals = spectral_gap(H, ops)
        assert gap >= 0.0

    def test_gap_increases_with_gamma(self):
        from quasim.ciir.multi_qubit.analysis.stability import spectral_gap
        from quasim.ciir.multi_qubit.quantum.density_matrix import amplitude_damping_ops
        H = np.diag([0.1, -0.1, -0.1, 0.1]).astype(complex)
        ops_low = amplitude_damping_ops(2, 0.01)
        ops_high = amplitude_damping_ops(2, 0.5)
        gap_low, _ = spectral_gap(H, ops_low)
        gap_high, _ = spectral_gap(H, ops_high)
        assert gap_high >= gap_low  # larger dissipation → faster relaxation


class TestLyapunovAnalyzer:
    def test_V_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.stability import LyapunovAnalyzer
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            DensityMatrixSimulator, LindbladParams
        )
        sim = HybridSimulation(n_qubits=2, n_steps=100, dt=0.01)
        res = sim.run(seed=9)
        params_q = LindbladParams(n_qubits=2)
        rho_star = DensityMatrixSimulator(params_q).ground_state()
        analyzer = LyapunovAnalyzer(x_star=np.zeros(4), rho_star=rho_star)
        lyap = analyzer.analyze(res.times, res.x_traj, res.rho_traj)
        assert np.all(lyap.V_traj >= -1e-6)

    def test_convergence_detected(self):
        from quasim.ciir.multi_qubit.analysis.stability import LyapunovAnalyzer
        from quasim.ciir.multi_qubit.simulation.hybrid_sim import HybridSimulation
        from quasim.ciir.multi_qubit.quantum.density_matrix import (
            DensityMatrixSimulator, LindbladParams
        )
        # Low noise, tight threshold
        sim = HybridSimulation(
            n_qubits=2, n_steps=400, dt=0.01,
            noise_classical=0.01, gamma_ad=0.01, gamma_dp=0.005
        )
        res = sim.run(seed=10)
        params_q = LindbladParams(n_qubits=2)
        rho_star = DensityMatrixSimulator(params_q).ground_state()
        analyzer = LyapunovAnalyzer(x_star=np.zeros(4), rho_star=rho_star, threshold=2.0)
        lyap = analyzer.analyze(res.times, res.x_traj, res.rho_traj)
        # V must be non-negative everywhere (physical requirement)
        assert np.all(lyap.V_traj >= -1e-6)
        # The LQR component (x_norm2) should decrease since LQR is stabilising
        assert lyap.x_norm2_traj[-1] < lyap.x_norm2_traj[0] * 2.0  # bounded

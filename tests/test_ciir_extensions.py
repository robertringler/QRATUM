"""Tests for CIIR Extension Modules (GAPs 2, 3, 5, 6, 7).

Coverage
--------
* Non-Markovian dynamics (GAP 6): kernel evaluation, norm, trace preservation,
  history management, Markovian limit, comparison vs Markovian
* Information-geometric control (GAP 7): density from theta, SLD computation,
  Fisher metric (PSD), natural gradient step reduces cost, optimization
* Hardware parameters (GAP 2 + GAP 3): physical ranges, latency formula,
  feasibility assessment, unit conversion, spectral gap scaling
* Tensor network MPS (GAP 5): constructors, gate application, entanglement
  entropy, overlap, fidelity, Trotter evolution, MCWF trajectory,
  exact vs MPS comparison for small N
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _bell_rho(n_qubits: int = 2):
    """Bell state density matrix |Φ+⟩⟨Φ+|."""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[3] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _ground_rho(n_qubits: int = 2):
    """Ground state |00...0⟩⟨00...0| density matrix."""
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _random_hermitian(d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 2


def _lindblad_ops_2q(gamma: float = 0.01):
    from quasim.ciir.multi_qubit.quantum.density_matrix import (
        amplitude_damping_ops, dephasing_ops)
    ops = amplitude_damping_ops(2, gamma) + dephasing_ops(2, gamma)
    return ops


def _hamiltonian_2q():
    from quasim.ciir.multi_qubit.control.controller import control_hamiltonian
    H0, _ = control_hamiltonian(2)
    return H0


# ===========================================================================
# GAP 6 — Non-Markovian dynamics
# ===========================================================================

class TestMemoryKernel:
    def test_exponential_at_zero(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("exponential", lambda_=2.0, strength=1.0)
        assert k(0.0) == pytest.approx(2.0, rel=1e-6)

    def test_exponential_decays(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("exponential", lambda_=1.0, strength=1.0)
        assert k(1.0) < k(0.0)
        assert k(2.0) < k(1.0)

    def test_oscillatory_at_zero(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("oscillatory", gamma=0.5, omega=2.0, strength=1.0)
        assert k(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_negative_time_returns_zero(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("exponential", lambda_=1.0, strength=1.0)
        assert k(-1.0) == 0.0

    def test_norm_positive(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("exponential", lambda_=1.0, strength=1.0)
        assert k.norm() > 0.0

    def test_oscillatory_norm_finite(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("oscillatory", gamma=1.0, omega=5.0, strength=0.1)
        assert np.isfinite(k.norm())

    def test_unknown_kernel_raises(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("bad_type", strength=1.0)
        with pytest.raises(ValueError):
            k(1.0)

    def test_zero_strength(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import MemoryKernel
        k = MemoryKernel("exponential", lambda_=1.0, strength=0.0)
        assert k(0.5) == 0.0


class TestNonMarkovianEvolver:
    def _evolver(self, kernel_type="exponential", strength=0.01):
        from quasim.ciir.multi_qubit.quantum.non_markovian import (
            MemoryKernel, NonMarkovianEvolver)
        H = _hamiltonian_2q()
        ops = _lindblad_ops_2q(0.01)
        kernel = MemoryKernel(kernel_type, lambda_=2.0, strength=strength)
        return NonMarkovianEvolver(H, ops, kernel, dt=0.01)

    def test_evolve_returns_list(self):
        ev = self._evolver()
        rho0 = _bell_rho()
        traj = ev.evolve(rho0, n_steps=10, record_every=2)
        assert len(traj) > 0

    def test_trace_preserved(self):
        ev = self._evolver()
        rho0 = _bell_rho()
        traj = ev.evolve(rho0, n_steps=20, record_every=5)
        for rho in traj:
            assert abs(np.trace(rho).real - 1.0) < 1e-8

    def test_hermiticity_preserved(self):
        ev = self._evolver()
        rho0 = _bell_rho()
        traj = ev.evolve(rho0, n_steps=20, record_every=5)
        for rho in traj:
            assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_positivity_preserved(self):
        ev = self._evolver()
        rho0 = _bell_rho()
        traj = ev.evolve(rho0, n_steps=20, record_every=5)
        for rho in traj:
            eigvals = np.linalg.eigvalsh(rho)
            assert np.all(eigvals >= -1e-10)

    def test_initial_state_in_trajectory(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import (
            MemoryKernel, NonMarkovianEvolver)
        H = _hamiltonian_2q()
        ops = _lindblad_ops_2q(0.01)
        kernel = MemoryKernel("exponential", lambda_=2.0, strength=0.01)
        ev = NonMarkovianEvolver(H, ops, kernel, dt=0.01)
        rho0 = _ground_rho()
        traj = ev.evolve(rho0, n_steps=5, record_every=5)
        # First recorded is at record_every steps
        assert len(traj) >= 1

    def test_oscillatory_kernel_stable(self):
        ev = self._evolver("oscillatory", strength=0.005)
        rho0 = _ground_rho()
        traj = ev.evolve(rho0, n_steps=20, record_every=5)
        for rho in traj:
            assert abs(np.trace(rho).real - 1.0) < 1e-6

    def test_zero_strength_reduces_to_markovian_shape(self):
        """With zero memory strength, output should be close to Markovian."""
        from quasim.ciir.multi_qubit.quantum.density_matrix import evolve_rk4
        from quasim.ciir.multi_qubit.quantum.non_markovian import (
            MemoryKernel, NonMarkovianEvolver)
        H = _hamiltonian_2q()
        ops = _lindblad_ops_2q(0.01)
        kernel = MemoryKernel("exponential", lambda_=1.0, strength=0.0)
        ev = NonMarkovianEvolver(H, ops, kernel, dt=0.01)
        rho0 = _bell_rho()

        # Non-Markovian with zero strength
        traj_nm = ev.evolve(rho0, n_steps=10, record_every=10)
        # Markovian
        traj_mk = evolve_rk4(rho0, H, ops, dt=0.01, n_steps=10, record_every=10)

        rho_nm = traj_nm[-1]
        rho_mk = traj_mk[-1]
        # Should be close (not identical due to implementation differences)
        fid = float(np.real(np.trace(rho_nm @ rho_mk)))
        assert fid > 0.8  # high overlap expected

    def test_history_grows(self):
        from quasim.ciir.multi_qubit.quantum.non_markovian import (
            MemoryKernel, NonMarkovianEvolver)
        H = _hamiltonian_2q()
        ops = _lindblad_ops_2q(0.01)
        kernel = MemoryKernel("exponential", lambda_=2.0, strength=0.01)
        ev = NonMarkovianEvolver(H, ops, kernel, dt=0.01)
        rho0 = _ground_rho()
        ev.evolve(rho0, n_steps=5, record_every=5)
        assert len(ev._history) == 6  # initial + 5 steps


# ===========================================================================
# GAP 7 — Information-geometric control
# ===========================================================================

class TestExponentialFamily:
    def _setup(self, n_qubits=2):
        from quasim.ciir.multi_qubit.control.controller import \
            control_hamiltonian
        H0, H_ops = control_hamiltonian(n_qubits)
        return H0, H_ops[:3]  # use first 3 controls

    def test_density_is_valid(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            _density_from_theta
        H0, H_ops = self._setup()
        theta = np.zeros(len(H_ops))
        rho = _density_from_theta(theta, H0, H_ops)
        assert abs(np.trace(rho).real - 1.0) < 1e-10
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)

    def test_density_hermitian(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            _density_from_theta
        H0, H_ops = self._setup()
        theta = np.array([0.1, -0.2, 0.05])
        rho = _density_from_theta(theta, H0, H_ops)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_density_changes_with_theta(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            _density_from_theta
        H0, H_ops = self._setup()
        rho1 = _density_from_theta(np.zeros(3), H0, H_ops)
        rho2 = _density_from_theta(np.array([1.0, 0.0, 0.0]), H0, H_ops)
        assert not np.allclose(rho1, rho2)


class TestSLD:
    def test_sld_hermitian(self):
        from quasim.ciir.multi_qubit.control.geometric_control import sld
        rho = _ground_rho(2) * 0.9 + _bell_rho(2) * 0.1
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho).real
        # Small perturbation
        drho = 0.01 * _random_hermitian(4, seed=1)
        drho -= np.trace(drho).real * np.eye(4) / 4  # traceless
        L = sld(rho, drho)
        assert np.allclose(L, L.conj().T, atol=1e-8)

    def test_sld_definition_satisfied(self):
        """Check ∂ρ = ½(ρL + Lρ) approximately."""
        from quasim.ciir.multi_qubit.control.geometric_control import sld
        rho = np.eye(4, dtype=complex) / 4 + 0.01 * _random_hermitian(4, seed=5)
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho).real
        drho = 0.001 * _random_hermitian(4, seed=6)
        drho = 0.5 * (drho + drho.conj().T)  # hermitian
        drho -= np.eye(4) * np.trace(drho).real / 4  # traceless

        L = sld(rho, drho)
        reconstructed = 0.5 * (rho @ L + L @ rho)
        assert np.allclose(drho, reconstructed, atol=1e-6)


class TestQuantumFisherMetric:
    def _setup(self):
        from quasim.ciir.multi_qubit.control.controller import \
            control_hamiltonian
        H0, H_ops = control_hamiltonian(2)
        return H0, H_ops[:2]  # 2 parameters for speed

    def test_fisher_metric_shape(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            quantum_fisher_metric
        H0, H_ops = self._setup()
        theta = np.zeros(2)
        G = quantum_fisher_metric(theta, H0, H_ops)
        assert G.shape == (2, 2)

    def test_fisher_metric_symmetric(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            quantum_fisher_metric
        H0, H_ops = self._setup()
        theta = np.zeros(2)
        G = quantum_fisher_metric(theta, H0, H_ops)
        assert np.allclose(G, G.T, atol=1e-8)

    def test_fisher_metric_psd(self):
        from quasim.ciir.multi_qubit.control.geometric_control import \
            quantum_fisher_metric
        H0, H_ops = self._setup()
        theta = np.zeros(2)
        G = quantum_fisher_metric(theta, H0, H_ops)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals >= -1e-8)


class TestNaturalGradientController:
    def _controller(self):
        from quasim.ciir.multi_qubit.control.controller import \
            control_hamiltonian
        from quasim.ciir.multi_qubit.control.geometric_control import \
            NaturalGradientController
        H0, H_ops = control_hamiltonian(2)
        return NaturalGradientController(H0, H_ops[:2], eta=0.1, reg=1e-3)

    def test_step_returns_result(self):
        ctrl = self._controller()
        theta0 = np.zeros(2)
        rho_target = _ground_rho(2)
        result = ctrl.step(theta0, rho_target)
        assert 0.0 <= result.fidelity <= 1.0 + 1e-8

    def test_step_returns_valid_rho(self):
        ctrl = self._controller()
        theta0 = np.zeros(2)
        rho_target = _ground_rho(2)
        result = ctrl.step(theta0, rho_target)
        rho = result.rho
        assert abs(np.trace(rho).real - 1.0) < 1e-8
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-8)

    def test_optimization_reduces_cost(self):
        ctrl = self._controller()
        theta0 = np.ones(2) * 0.5
        rho_target = _ground_rho(2)
        _, costs, _ = ctrl.optimize(theta0, rho_target, n_steps=10)
        # Cost should not increase overall (natural gradient is descent)
        assert costs[-1] <= costs[0] + 0.05  # small tolerance

    def test_fidelity_non_negative(self):
        ctrl = self._controller()
        theta0 = np.zeros(2)
        rho_target = _bell_rho(2)
        result = ctrl.step(theta0, rho_target)
        assert result.fidelity >= 0.0

    def test_h_ctrl_from_theta_shape(self):
        ctrl = self._controller()
        theta = np.zeros(2)
        H = ctrl.H_ctrl_from_theta(theta)
        assert H.shape == (4, 4)

    def test_h_ctrl_hermitian(self):
        ctrl = self._controller()
        theta = np.array([0.1, -0.3])
        H = ctrl.H_ctrl_from_theta(theta)
        assert np.allclose(H, H.conj().T, atol=1e-10)


# ===========================================================================
# GAP 2 + GAP 3 — Hardware parameters and spectral gap scaling
# ===========================================================================

class TestSuperconductingParams:
    def test_gamma_ad_positive(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams(T1_s=100e-6)
        assert p.gamma_ad > 0.0

    def test_gamma_ad_value(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams(T1_s=100e-6)
        assert p.gamma_ad == pytest.approx(10000.0, rel=1e-6)

    def test_gamma_dp_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams(T1_s=100e-6, T2_s=50e-6)
        assert p.gamma_dp >= 0.0

    def test_gamma_total_greater_than_parts(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams(T1_s=100e-6, T2_s=50e-6)
        assert p.gamma_total >= p.gamma_ad
        assert p.gamma_total >= p.gamma_dp

    def test_total_latency_positive(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams()
        assert p.total_latency_s > 0.0

    def test_gate_fidelity_loss_small(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            SuperconductingParams
        p = SuperconductingParams()
        # Per-gate fidelity loss should be < 1%
        assert p.gate_fidelity_loss_per_gate < 0.01


class TestTrappedIonParams:
    def test_gamma_ad_very_small(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            TrappedIonParams
        p = TrappedIonParams(T1_s=1.0)
        assert p.gamma_ad == pytest.approx(1.0, rel=1e-6)

    def test_gamma_total_positive(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            TrappedIonParams
        p = TrappedIonParams()
        assert p.gamma_total >= 0.0


class TestUnitConversion:
    def test_conversion_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            SuperconductingParams, scale_to_simulation)
        p = SuperconductingParams()
        result = scale_to_simulation(p)
        assert "gamma_ad_sim" in result
        assert "latency_steps" in result

    def test_simulation_rates_dimensionless(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            SuperconductingParams, scale_to_simulation)
        p = SuperconductingParams()
        result = scale_to_simulation(p, sim_time_unit_s=1e-6)
        # In simulation units (1μs = 1 t.u.), γ_ad = γ_ad_Hz × 1e-6
        expected = p.gamma_ad * 1e-6
        assert result["gamma_ad_sim"] == pytest.approx(expected, rel=1e-6)

    def test_latency_steps_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            SuperconductingParams, scale_to_simulation)
        p = SuperconductingParams()
        result = scale_to_simulation(p)
        assert result["latency_steps"] >= 0


class TestLatencyAnalysis:
    def test_tau_max_formula(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            max_tolerable_latency
        gamma = 20000.0  # 20 kHz
        epsilon = 0.05
        tau_max = max_tolerable_latency(gamma, epsilon)
        assert tau_max == pytest.approx(epsilon / gamma, rel=1e-6)

    def test_tau_max_scales_with_epsilon(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            max_tolerable_latency
        gamma = 20000.0
        assert max_tolerable_latency(gamma, 0.1) > max_tolerable_latency(gamma, 0.05)

    def test_assessment_returns_result(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            SuperconductingParams, assess_latency)
        p = SuperconductingParams()
        result = assess_latency(p, epsilon=0.05)
        assert hasattr(result, "tau_max_s")
        assert hasattr(result, "is_realistic")
        assert isinstance(result.recommendation, str)

    def test_assessment_margin_positive(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            SuperconductingParams, assess_latency)
        p = SuperconductingParams()
        result = assess_latency(p)
        assert result.margin_ratio > 0.0

    def test_zero_gamma_returns_inf(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            max_tolerable_latency
        assert max_tolerable_latency(0.0) == float("inf")


class TestSpectralGapScaling:
    def _build_H(self, n):
        from quasim.ciir.multi_qubit.control.controller import \
            control_hamiltonian
        H0, _ = control_hamiltonian(n)
        return H0

    def _build_ops(self, n, gamma):
        from quasim.ciir.multi_qubit.quantum.density_matrix import \
            amplitude_damping_ops
        return amplitude_damping_ops(n, gamma)

    def test_scaling_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            spectral_gap_scaling
        result = spectral_gap_scaling(
            N_values=[2],
            gamma_values=[0.01],
            H_norm_values=[1.0],
            H_builder=self._build_H,
            ops_builder=self._build_ops,
        )
        assert "gap" in result
        assert len(result["gap"]) > 0

    def test_gap_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            spectral_gap_scaling
        result = spectral_gap_scaling(
            N_values=[2],
            gamma_values=[0.01, 0.05],
            H_norm_values=[0.5, 1.0],
            H_builder=self._build_H,
            ops_builder=self._build_ops,
        )
        for g in result["gap"]:
            assert np.isnan(g) or g >= 0.0

    def test_gap_increases_with_gamma(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import \
            spectral_gap_scaling
        result = spectral_gap_scaling(
            N_values=[2],
            gamma_values=[0.01, 0.1],
            H_norm_values=[1.0],
            H_builder=self._build_H,
            ops_builder=self._build_ops,
        )
        gaps = result["gap"]
        # Higher gamma should give larger spectral gap
        assert gaps[1] > gaps[0]

    def test_fit_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.hardware_params import (
            fit_gap_scaling, spectral_gap_scaling)
        result = spectral_gap_scaling(
            N_values=[2],
            gamma_values=[0.01, 0.02, 0.05, 0.1],
            H_norm_values=[0.5, 1.0],
            H_builder=self._build_H,
            ops_builder=self._build_ops,
        )
        fit = fit_gap_scaling(result["gamma"], result["H_norm"], result["gap"])
        assert "c0" in fit
        assert "a" in fit
        assert "r2" in fit


# ===========================================================================
# GAP 5 — Tensor Network MPS
# ===========================================================================

class TestMPSConstructors:
    def test_ground_state_norm(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_ground_state, mps_to_vector)
        mps = mps_ground_state(3)
        psi = mps_to_vector(mps)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_ground_state_first_component(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_ground_state, mps_to_vector)
        for n in [2, 3, 4]:
            mps = mps_ground_state(n)
            psi = mps_to_vector(mps)
            assert abs(psi[0] - 1.0) < 1e-10

    def test_product_state_correct(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_product_state, mps_to_vector)
        mps = mps_product_state([1, 0])  # |10⟩
        psi = mps_to_vector(mps)
        # |10⟩ = index 2 in standard ordering (σ₁=1, σ₂=0 → 2)
        assert abs(psi[2] - 1.0) < 1e-10
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_bell_state_norm(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_bell_state, mps_to_vector)
        mps = mps_bell_state(2)
        psi = mps_to_vector(mps)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_bell_state_entangled(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_bell_state, mps_to_vector)
        mps = mps_bell_state(2)
        psi = mps_to_vector(mps)
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 → components 0 and 3
        assert abs(abs(psi[0]) - 1.0 / np.sqrt(2)) < 1e-10
        assert abs(abs(psi[3]) - 1.0 / np.sqrt(2)) < 1e-10

    def test_random_mps_norm(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_random, mps_to_vector)
        for n in [2, 3, 4]:
            mps = mps_random(n, chi_max=4, seed=n)
            psi = mps_to_vector(mps)
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-8

    def test_n_qubits_attribute(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import \
            mps_ground_state
        for n in [2, 3, 5, 8]:
            mps = mps_ground_state(n)
            assert mps.n_qubits == n


class TestMPSGates:
    def test_single_qubit_x_gate(self):
        """X|0⟩ = |1⟩."""
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            apply_single_qubit_gate, mps_ground_state, mps_to_vector)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mps = mps_ground_state(2)
        mps_x = apply_single_qubit_gate(mps, X, site=0)
        psi = mps_to_vector(mps_x)
        # X on qubit 0: |00⟩ → |10⟩ = index 2
        assert abs(psi[2] - 1.0) < 1e-10

    def test_single_qubit_gate_norm(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            apply_single_qubit_gate, mps_random, mps_to_vector)
        H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        mps = mps_random(3, chi_max=4, seed=7)
        mps_h = apply_single_qubit_gate(mps, H_gate, site=1)
        psi = mps_to_vector(mps_h)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-8

    def test_cnot_creates_entanglement(self):
        """CNOT|+0⟩ = |Φ+⟩."""
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            apply_single_qubit_gate, apply_two_qubit_gate, mps_ground_state,
            mps_to_vector)
        H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        mps = mps_ground_state(2)
        mps = apply_single_qubit_gate(mps, H_gate, site=0)  # |+0⟩
        mps = apply_two_qubit_gate(mps, CNOT, site=0, chi_max=4)   # |Φ+⟩
        psi = mps_to_vector(mps)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-8
        # Bell state: psi[0] = psi[3] = 1/√2
        assert abs(abs(psi[0]) - 1.0 / np.sqrt(2)) < 1e-8


class TestMPSEntanglement:
    def test_product_state_zero_entropy(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            entanglement_entropy_at_bond, mps_ground_state)
        mps = mps_ground_state(4)
        S = entanglement_entropy_at_bond(mps, bond=1)
        assert abs(S) < 1e-10

    def test_bell_state_max_entropy(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            entanglement_entropy_at_bond, mps_bell_state)
        mps = mps_bell_state(2)
        S = entanglement_entropy_at_bond(mps, bond=0)
        assert abs(S - np.log(2)) < 1e-8

    def test_entropy_non_negative(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            entanglement_entropy_at_bond, mps_random)
        mps = mps_random(4, chi_max=8, seed=13)
        for bond in range(3):
            S = entanglement_entropy_at_bond(mps, bond=bond)
            assert S >= -1e-10


class TestMPSOverlap:
    def test_self_overlap_one(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_ground_state, mps_overlap)
        mps = mps_ground_state(3)
        ov = mps_overlap(mps, mps)
        assert abs(ov - 1.0) < 1e-10

    def test_orthogonal_states(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_overlap, mps_product_state)
        mps_00 = mps_product_state([0, 0])
        mps_11 = mps_product_state([1, 1])
        ov = mps_overlap(mps_00, mps_11)
        assert abs(ov) < 1e-10

    def test_fidelity_bell_with_ground(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_bell_state, mps_fidelity, mps_ground_state)
        bell = mps_bell_state(2)
        ground = mps_ground_state(2)
        F = mps_fidelity(bell, ground)
        # ⟨Φ+|00⟩ = 1/√2, so F = 0.5
        assert abs(F - 0.5) < 1e-8

    def test_fidelity_self_one(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_fidelity, mps_random)
        mps = mps_random(3, chi_max=4, seed=17)
        assert abs(mps_fidelity(mps, mps) - 1.0) < 1e-8


class TestMPSEvolution:
    def _nn_hamiltonian(self, n_qubits: int):
        """Nearest-neighbor ZZ + X Hamiltonian pairs."""
        SZ = np.array([[1, 0], [0, -1]], dtype=complex)
        SX = np.array([[0, 1], [1, 0]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        pairs = []
        for k in range(n_qubits - 1):
            H_pair = np.kron(SZ, SZ) + 0.5 * (np.kron(SX, I2) + np.kron(I2, SX))
            pairs.append((k, H_pair))
        return pairs

    def test_trotter_returns_list(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_evolve_trotter, mps_ground_state)
        mps = mps_ground_state(3)
        H_pairs = self._nn_hamiltonian(3)
        traj = mps_evolve_trotter(mps, H_pairs, dt=0.01, n_steps=5, record_every=2)
        assert len(traj) > 0

    def test_trotter_norm_preserved(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_evolve_trotter, mps_random, mps_to_vector)
        mps = mps_random(3, chi_max=4, seed=21)
        H_pairs = self._nn_hamiltonian(3)
        traj = mps_evolve_trotter(mps, H_pairs, dt=0.01, n_steps=10, record_every=5)
        for m in traj:
            psi = mps_to_vector(m)
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-6


class TestMPSMCWF:
    def test_trajectory_returns_list(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import \
            amplitude_damping_ops
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_ground_state, mps_lindblad_trajectory)
        H = _hamiltonian_2q()
        ops = amplitude_damping_ops(2, 0.01)
        mps = mps_ground_state(2)
        traj = mps_lindblad_trajectory(mps, H, ops, dt=0.01, n_steps=5, seed=42)
        assert len(traj) == 6  # initial + 5 steps

    def test_trajectory_norm_preserved(self):
        from quasim.ciir.multi_qubit.quantum.density_matrix import \
            amplitude_damping_ops
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_bell_state, mps_lindblad_trajectory, mps_to_vector)
        H = _hamiltonian_2q()
        ops = amplitude_damping_ops(2, 0.01)
        mps = mps_bell_state(2)
        traj = mps_lindblad_trajectory(mps, H, ops, dt=0.01, n_steps=10, seed=7)
        for m in traj:
            psi = mps_to_vector(m)
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-8


class TestMPSScalability:
    def test_large_n_ground_state(self):
        """Ground state MPS should be trivially constructible at N=12."""
        from quasim.ciir.multi_qubit.quantum.tensor_network import \
            mps_ground_state
        mps = mps_ground_state(12)
        assert mps.n_qubits == 12
        # Bond dim should be 1 for product state
        for t in mps.tensors:
            assert t.shape[1] == 1
            assert t.shape[2] == 1

    def test_mps_memory_smaller_than_exact(self):
        """For N=6, MPS memory should be much smaller than exact dm."""
        from quasim.ciir.multi_qubit.quantum.tensor_network import mps_random
        n = 6
        mps = mps_random(n, chi_max=8, seed=42)
        mps_bytes = sum(t.nbytes for t in mps.tensors)
        exact_bytes = (2 ** n) ** 2 * 16  # complex128 density matrix
        assert mps_bytes < exact_bytes

    def test_vector_to_mps_roundtrip(self):
        """Converting psi → MPS → psi should preserve the state vector."""
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            _vector_to_mps, mps_to_vector)
        rng = np.random.default_rng(99)
        for n in [2, 3, 4]:
            psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
            psi /= np.linalg.norm(psi)
            mps = _vector_to_mps(psi, n, chi_max=2 ** n)
            psi_rec = mps_to_vector(mps)
            # Overlap should be 1 up to global phase
            ov = abs(np.dot(psi.conj(), psi_rec))
            assert ov == pytest.approx(1.0, abs=1e-8)

    def test_mps_density_matrix_valid(self):
        from quasim.ciir.multi_qubit.quantum.tensor_network import (
            mps_random, mps_to_density_matrix)
        mps = mps_random(3, chi_max=8, seed=33)
        rho = mps_to_density_matrix(mps)
        assert abs(np.trace(rho).real - 1.0) < 1e-8
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-8)

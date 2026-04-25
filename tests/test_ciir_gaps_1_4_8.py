"""Tests for CIIR GAPs 1, 4, and 8.

Coverage
--------
* alpha_derivation.py (GAP 1): spectral route, Hausdorff route, range checks
* causal_claim.py (GAP 4): simulation, observable computation, Ramsey protocol
* unified_evolution.py (GAP 8): runs without error, fidelity in [0,1],
  returns expected keys
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ground_rho(d: int = 4):
    """Ground state density matrix for Hilbert space of dimension d."""
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _mixed_rho(d: int = 4):
    """Maximally mixed density matrix."""
    return np.eye(d, dtype=complex) / d


def _simple_hamiltonian(d: int = 4):
    """Simple Hermitian Hamiltonian (d × d)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 4.0


def _simple_lindblad_ops(d: int = 4, gamma: float = 0.02):
    """Single amplitude-damping-like Lindblad operator."""
    L = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        L[i, i + 1] = np.sqrt(gamma)
    return [L]


# ===========================================================================
# GAP 1 — alpha_derivation.py
# ===========================================================================

class TestAlphaSpectralRoute:
    def _system(self, d: int = 4):
        H = _simple_hamiltonian(d)
        ops = _simple_lindblad_ops(d, gamma=0.02)
        return H, ops

    def test_returns_float(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_spectral
        H, ops = self._system()
        alpha, info = derive_alpha_spectral(H, ops)
        assert isinstance(alpha, float)

    def test_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_spectral
        H, ops = self._system()
        _, info = derive_alpha_spectral(H, ops)
        assert isinstance(info, dict)
        assert "alpha" in info
        assert "method" in info

    def test_alpha_positive(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_spectral
        H, ops = self._system()
        alpha, _ = derive_alpha_spectral(H, ops)
        assert alpha > 0.0

    def test_alpha_in_broad_range(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import (
            derive_alpha_spectral, verify_alpha_in_range,
        )
        H, ops = self._system()
        alpha, _ = derive_alpha_spectral(H, ops)
        assert verify_alpha_in_range(alpha), f"α = {alpha:.4f} outside [1.5, 2.5]"

    def test_single_op_does_not_crash(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_spectral
        H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
        L = np.zeros((4, 4), dtype=complex)
        L[0, 1] = 0.1
        alpha, info = derive_alpha_spectral(H, [L])
        assert isinstance(alpha, float)

    def test_no_ops_fallback(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_spectral
        H = np.diag([0.0, 1.0]).astype(complex)
        alpha, info = derive_alpha_spectral(H, [])
        assert 1.0 <= alpha <= 3.0


class TestAlphaHausdorffRoute:
    def test_returns_float(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_hausdorff
        alpha, info = derive_alpha_hausdorff(n_copies=2.3, scale_factor=1.0 / 3.0)
        assert isinstance(alpha, float)

    def test_formula(self):
        """d_H = log(n_copies)/log(3), α = 1 + d_H."""
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_hausdorff
        n_copies = 3.0
        s = 1.0 / 3.0
        alpha, info = derive_alpha_hausdorff(n_copies, s)
        expected = 1.0 + np.log(3.0) / np.log(3.0)  # = 2.0
        assert alpha == pytest.approx(expected, rel=1e-6)

    def test_target_range_achievable(self):
        """n_copies = 3^0.83, s=1/3 should give α ≈ 1.83."""
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_hausdorff
        s = 1.0 / 3.0
        d_H_mid = 0.83
        n_copies = 3.0 ** d_H_mid
        alpha, _ = derive_alpha_hausdorff(n_copies, s)
        assert 1.5 <= alpha <= 2.5

    def test_invalid_scale_raises(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import derive_alpha_hausdorff
        with pytest.raises(ValueError):
            derive_alpha_hausdorff(2.0, scale_factor=1.5)

    def test_default_ifs_parameters(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import (
            default_ifs_parameters, derive_alpha_hausdorff,
        )
        n_copies, s = default_ifs_parameters()
        alpha, info = derive_alpha_hausdorff(n_copies, s)
        assert 1.0 < alpha < 3.0


class TestAlphaCrossValidation:
    def test_cross_validate_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import cross_validate_routes
        H = _simple_hamiltonian(4)
        ops = _simple_lindblad_ops(4)
        result = cross_validate_routes(H, ops)
        assert "alpha_spectral" in result
        assert "alpha_hausdorff" in result
        assert "agreement" in result

    def test_both_in_broad_range(self):
        from quasim.ciir.multi_qubit.analysis.alpha_derivation import cross_validate_routes
        H = _simple_hamiltonian(4)
        ops = _simple_lindblad_ops(4)
        result = cross_validate_routes(H, ops)
        assert result["both_in_broad_range"]


# ===========================================================================
# GAP 4 — causal_claim.py
# ===========================================================================

class TestCausalClaimSimulation:
    def _system(self):
        d = 4
        H = _simple_hamiltonian(d)
        ops = _simple_lindblad_ops(d, gamma=0.01)
        rho0 = _ground_rho(d)
        return H, ops, rho0, d

    def test_ciir_model_returns_list(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import simulate_ciir_model
        H, ops, rho0, d = self._system()
        traj = simulate_ciir_model(rho0, H, ops, n_steps=5)
        assert isinstance(traj, list)
        assert len(traj) == 6

    def test_standard_model_returns_list(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import simulate_standard_model
        H, ops, rho0, d = self._system()
        traj = simulate_standard_model(rho0, H, ops, n_steps=5)
        assert len(traj) == 6

    def test_trajectory_trace_one(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import simulate_ciir_model
        H, ops, rho0, d = self._system()
        traj = simulate_ciir_model(rho0, H, ops, n_steps=10)
        for rho in traj:
            assert float(np.trace(rho).real) == pytest.approx(1.0, abs=1e-6)

    def test_trajectory_hermitian(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import simulate_standard_model
        H, ops, rho0, d = self._system()
        traj = simulate_standard_model(rho0, H, ops, n_steps=5)
        for rho in traj:
            assert np.allclose(rho, rho.conj().T, atol=1e-8)

    def test_distinguishing_observable_structure(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            simulate_ciir_model, simulate_standard_model,
            compute_distinguishing_observable,
        )
        H, ops, rho0, d = self._system()
        ciir_traj = simulate_ciir_model(rho0, H, ops, n_steps=8, noise_strength=0.15)
        std_traj = simulate_standard_model(rho0, H, ops, n_steps=8, noise_strength=0.15)
        result = compute_distinguishing_observable(ciir_traj, std_traj, rho0)
        assert "fidelity_ciir" in result
        assert "fidelity_std" in result
        assert "delta_fidelity" in result
        assert "proposition_P_supported" in result
        assert "final_delta" in result

    def test_fidelity_in_range(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            simulate_ciir_model, simulate_standard_model,
            compute_distinguishing_observable,
        )
        H, ops, rho0, d = self._system()
        ciir_traj = simulate_ciir_model(rho0, H, ops, n_steps=5)
        std_traj = simulate_standard_model(rho0, H, ops, n_steps=5)
        result = compute_distinguishing_observable(ciir_traj, std_traj, rho0)
        assert all(0.0 <= f <= 1.0 for f in result["fidelity_ciir"])
        assert all(0.0 <= f <= 1.0 for f in result["fidelity_std"])


class TestRamseyProtocol:
    def test_design_returns_protocol(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import design_ramsey_protocol
        protocol = design_ramsey_protocol()
        assert protocol.tau_noise_steps >= 1
        assert protocol.tau_free_steps >= 1
        assert protocol.p_noise > 0

    def test_above_threshold(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import design_ramsey_protocol
        protocol = design_ramsey_protocol(p_noise=0.15, p_star=0.1)
        assert protocol.above_threshold()

    def test_run_ramsey_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            design_ramsey_protocol, run_ramsey_experiment,
        )
        d = 4
        H = _simple_hamiltonian(d)
        ops = _simple_lindblad_ops(d, gamma=0.01)
        rho0 = _ground_rho(d)
        protocol = design_ramsey_protocol(
            dt=0.05, tau_noise=0.1, tau_free=0.1, p_noise=0.15
        )
        result = run_ramsey_experiment(rho0, H, ops, protocol=protocol)
        assert "protocol" in result
        assert "observable" in result
        assert "ciir_final" in result
        assert "std_final" in result
        assert "above_threshold" in result

    def test_ramsey_final_states_are_density_matrices(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            design_ramsey_protocol, run_ramsey_experiment,
        )
        d = 4
        H = _simple_hamiltonian(d)
        ops = _simple_lindblad_ops(d, gamma=0.01)
        rho0 = _ground_rho(d)
        protocol = design_ramsey_protocol(dt=0.05, tau_noise=0.1, tau_free=0.1)
        result = run_ramsey_experiment(rho0, H, ops, protocol=protocol)
        for key in ("ciir_final", "std_final"):
            rho = result[key]
            assert float(np.trace(rho).real) == pytest.approx(1.0, abs=1e-5)
            assert np.allclose(rho, rho.conj().T, atol=1e-7)


# ===========================================================================
# GAP 8 — unified_evolution.py
# ===========================================================================

class TestUnifiedEvolver:
    def test_step_returns_density_matrix(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import (
            UnifiedCIIREvolver, _default_hamiltonian, _default_lindblad_ops,
        )
        N = 2
        dim = 2 ** N
        H = _default_hamiltonian(N)
        ops = _default_lindblad_ops(N, gamma=0.01)
        rho_target = np.zeros((dim, dim), dtype=complex)
        rho_target[0, 0] = 1.0
        rho0 = np.eye(dim, dtype=complex) / dim

        evolver = UnifiedCIIREvolver(H=H, ops=ops, rho_target=rho_target)
        rho1 = evolver.step(rho0, t=0.05)
        assert rho1.shape == (dim, dim)
        assert float(np.trace(rho1).real) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(rho1, rho1.conj().T, atol=1e-8)

    def test_evolve_trajectory_length(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import (
            UnifiedCIIREvolver, _default_hamiltonian, _default_lindblad_ops,
        )
        N = 2
        dim = 2 ** N
        H = _default_hamiltonian(N)
        ops = _default_lindblad_ops(N)
        rho_target = np.zeros((dim, dim), dtype=complex); rho_target[0, 0] = 1.0
        rho0 = np.eye(dim, dtype=complex) / dim
        evolver = UnifiedCIIREvolver(H=H, ops=ops, rho_target=rho_target)
        traj = evolver.evolve(rho0, n_steps=10)
        assert len(traj) == 11  # rho0 + 10 steps

    def test_fidelity_in_range(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import (
            UnifiedCIIREvolver, _default_hamiltonian, _default_lindblad_ops,
            _state_fidelity,
        )
        N = 2
        dim = 4
        H = _default_hamiltonian(N)
        ops = _default_lindblad_ops(N)
        rho_target = np.zeros((dim, dim), dtype=complex); rho_target[0, 0] = 1.0
        rho0 = np.eye(dim, dtype=complex) / dim
        evolver = UnifiedCIIREvolver(H=H, ops=ops, rho_target=rho_target)
        traj = evolver.evolve(rho0, n_steps=5)
        for rho in traj:
            fid = _state_fidelity(rho, rho_target)
            assert 0.0 <= fid <= 1.0


class TestRunUnifiedExperiment:
    def test_returns_expected_keys(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=5)
        expected_keys = {
            "fidelity_history", "fidelity_baseline",
            "control_energy", "memory_effect_strength",
            "hardware_latency_steps",
        }
        assert expected_keys.issubset(result.keys())

    def test_fidelity_history_length(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        n_steps = 8
        result = run_unified_experiment(N=2, n_steps=n_steps)
        assert len(result["fidelity_history"]) == n_steps + 1
        assert len(result["fidelity_baseline"]) == n_steps + 1

    def test_fidelity_values_in_range(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=6)
        for f in result["fidelity_history"]:
            assert 0.0 <= f <= 1.0, f"Fidelity {f} out of [0,1]"
        for f in result["fidelity_baseline"]:
            assert 0.0 <= f <= 1.0, f"Baseline fidelity {f} out of [0,1]"

    def test_control_energy_nonneg(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=5)
        assert result["control_energy"] >= 0.0

    def test_memory_effect_positive(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=5)
        assert result["memory_effect_strength"] > 0.0

    def test_hardware_latency_steps_integer(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=5)
        assert isinstance(result["hardware_latency_steps"], int)
        assert result["hardware_latency_steps"] >= 0

    def test_n3_runs_without_error(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=3, n_steps=4)
        assert "fidelity_history" in result

    def test_too_large_N_raises(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        with pytest.raises(ValueError):
            run_unified_experiment(N=13, n_steps=2)

    def test_baseline_vs_unified_both_valid(self):
        from quasim.ciir.multi_qubit.simulation.unified_evolution import run_unified_experiment
        result = run_unified_experiment(N=2, n_steps=5)
        # Both start and end fidelities should be in [0,1]
        assert 0.0 <= result["fidelity_history"][0] <= 1.0
        assert 0.0 <= result["fidelity_history"][-1] <= 1.0
        assert 0.0 <= result["fidelity_baseline"][0] <= 1.0
        assert 0.0 <= result["fidelity_baseline"][-1] <= 1.0

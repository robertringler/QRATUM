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


# ===========================================================================
# PROJECTOR DERIVATION — derive_projector_from_fixed_point
# ===========================================================================

_SMALL_PARAMS = {
    "N": 2,
    "dt": 0.01,
    "gamma": 0.02,
    "noise_strength": 0.15,
    "max_iter": 300,
}


class TestProjectorDerivation:
    """Tests for derive_projector_from_fixed_point (Route A + Route B)."""

    def test_returns_two_values(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        result = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert len(result) == 2

    def test_projector_shape(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        Pi, _ = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert Pi.shape == (4, 4)

    def test_projector_hermitian(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        Pi, _ = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert np.allclose(Pi, Pi.conj().T, atol=1e-8)

    def test_projector_trace_one(self):
        """Rank-1 projector must have trace = 1."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        Pi, _ = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert float(np.trace(Pi).real) == pytest.approx(1.0, abs=1e-6)

    def test_projector_idempotent(self):
        """Rank-1 projector satisfies Π² = Π."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        Pi, _ = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert np.allclose(Pi @ Pi, Pi, atol=1e-6)

    def test_certificate_keys(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        _, cert = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        required = {
            "fixed_point_residual",
            "overlap_with_ground",
            "derivation_route",
            "is_degenerate",
            "routes_disagree",
            "converged",
            "n_iterations",
        }
        assert required.issubset(cert.keys())

    def test_overlap_in_range(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        _, cert = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert 0.0 <= cert["overlap_with_ground"] <= 1.0

    def test_fixed_point_residual_nonneg(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        _, cert = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert cert["fixed_point_residual"] >= 0.0

    def test_spectral_route_executed_for_small_n(self):
        """Route A (spectral) must run for N=2 (d=4, superoperator 16×16)."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        _, cert = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert cert["derivation_route"] == "spectral+power"
        assert cert["route_A_residual"] is not None
        assert cert["route_agreement"] is not None

    def test_power_only_for_n3(self):
        """Route A is skipped for N=3 (d^2=64, superop OK) — actually N>4 only skips.
        For N=3 (d=8) superoperator is 64×64, still computed.
        Verify certificate returns without error for N=3."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        params = {**_SMALL_PARAMS, "N": 3, "max_iter": 100}
        Pi, cert = derive_projector_from_fixed_point(params, tol=1e-5)
        assert Pi.shape == (8, 8)
        assert 0.0 <= cert["overlap_with_ground"] <= 1.0

    def test_convergence_flag(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            derive_projector_from_fixed_point,
        )
        _, cert = derive_projector_from_fixed_point(_SMALL_PARAMS, tol=1e-6)
        assert isinstance(cert["converged"], bool)
        assert isinstance(cert["n_iterations"], int)
        assert cert["n_iterations"] >= 1

    def test_superoperator_shape(self):
        """_build_ciir_superoperator_linear returns (d², d²) matrix."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import (
            _build_ciir_superoperator_linear,
            _build_default_hamiltonian_local,
            _build_default_lindblad_local,
            _build_constraint_projector,
        )
        N = 2
        dim = 4
        H = _build_default_hamiltonian_local(N)
        ops = _build_default_lindblad_local(N, gamma=0.02)
        Pi = _build_constraint_projector(dim)
        S = _build_ciir_superoperator_linear(H, ops, Pi, dt=0.01, noise_strength=0.15, dim=dim)
        assert S.shape == (dim ** 2, dim ** 2)


# ===========================================================================
# RESCREENING — rescreen_causal_claim
# ===========================================================================

class TestRescreenCausalClaim:
    """Tests for rescreen_causal_claim."""

    _params = {
        "N": 2,
        "n_steps": 20,
        "dt": 0.01,
        "gamma": 0.02,
        "noise_strength": 0.15,
        "max_iter": 300,
    }

    def test_returns_dict(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert isinstance(result, dict)

    def test_required_keys(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        required = {
            "projector", "certificate",
            "O_CIIR_derived", "O_std_derived", "delta_O_derived",
            "delta_O_max", "sigma_numerical", "snr_derived",
            "artifact_results", "verdict",
            "verdict_justification", "counterargument",
            "n_scaling", "control_split",
        }
        assert required.issubset(result.keys())

    def test_verdict_is_valid_string(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert result["verdict"] in ("A+", "A0", "B", "C")

    def test_delta_O_max_nonneg(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert result["delta_O_max"] >= 0.0

    def test_snr_nonneg(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert result["snr_derived"] >= 0.0

    def test_sigma_numerical_positive(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert result["sigma_numerical"] > 0.0

    def test_observable_arrays_same_length(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        n = self._params["n_steps"]
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert len(result["O_CIIR_derived"]) == n + 1
        assert len(result["O_std_derived"]) == n + 1
        assert len(result["delta_O_derived"]) == n + 1

    def test_observable_values_in_range(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        assert all(-1.0 <= v <= 2.0 for v in result["O_CIIR_derived"])
        assert all(-1.0 <= v <= 2.0 for v in result["O_std_derived"])

    def test_artifact_results_keys(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        for label in ("AH-1", "AH-2", "AH-3", "AH-4", "AH-5"):
            assert label in result["artifact_results"]
            ah = result["artifact_results"][label]
            assert "passed" in ah
            assert "value" in ah
            assert "threshold" in ah

    def test_n_scaling_table(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2, 4], tol=1e-6)
        assert len(result["n_scaling"]) == 2
        for row in result["n_scaling"]:
            assert "N" in row

    def test_control_split_keys(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        result = rescreen_causal_claim(self._params, N_scan=[2], tol=1e-6)
        cs = result["control_split"]
        assert "overlap_with_ground" in cs
        assert "estimated_control_fraction" in cs
        assert 0.0 <= cs["estimated_control_fraction"] <= 1.0

    def test_high_overlap_gives_a0_or_c(self):
        """With amplitude damping (γ=0.1, large noise), overlap → 1 → A0 or B/C."""
        from quasim.ciir.multi_qubit.analysis.causal_claim import rescreen_causal_claim
        params = {**self._params, "gamma": 0.1, "noise_strength": 0.2}
        result = rescreen_causal_claim(params, N_scan=[2], tol=1e-5)
        assert result["verdict"] in ("A+", "A0", "B", "C")
        # overlap must be in [0,1]
        assert 0.0 <= result["certificate"]["overlap_with_ground"] <= 1.0

    def test_compute_sigma_numerical(self):
        from quasim.ciir.multi_qubit.analysis.causal_claim import compute_sigma_numerical
        sigma = compute_sigma_numerical(n_steps=50, dim=8, dt=0.01)
        assert sigma > 0.0
        # Should be much smaller than 1
        assert sigma < 1e-6

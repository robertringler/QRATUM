"""Tests for the CIIR Falsification Protocol (falsification.py).

Coverage
--------
* sigma_numerical: dimensionally consistent, positive, small
* run_ciir_trajectory: returns valid density matrices + fidelity in [0,1]
* run_null_trajectory: same, model B has lower fidelity than model A
* compute_signal_metrics: ΔO_max > 0, SNR > 0, arrays consistent
* test_ah1_bond_dimension: returns ArtifactResult with PASS/FAIL
* test_ah2_step_size: PASS under default parameters
* test_ah3_initial_state: PASS for small perturbation
* test_ah4_noise_sensitivity: returns valid result
* test_ah5_n_scaling: non-decreasing for N=2,4,6
* classify_verdict: correct letter for given SNR/artifact inputs
* run_falsification_protocol: end-to-end smoke test (N=2, fast)
* S1/S2/S3 secondary objectives: run without error, return expected keys
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_SMALL = 2  # dim = 4 — fast tests
N_STEPS = 20
DT = 0.02
GAMMA = 0.02
NOISE = 0.15


def _build(N=N_SMALL):
    from quasim.ciir.multi_qubit.analysis.falsification import (
        _build_hamiltonian,
        _build_initial_state,
        _build_lindblad_ops,
    )

    dim = 2**N
    H = _build_hamiltonian(N)
    ops = _build_lindblad_ops(N, GAMMA)
    rho0 = _build_initial_state(dim)
    return H, ops, rho0, dim


# ===========================================================================
# σ_numerical
# ===========================================================================


class TestSigmaNumerical:
    def test_positive(self):
        from quasim.ciir.multi_qubit.analysis.falsification import compute_sigma_numerical

        s = compute_sigma_numerical(50, 4, 0.01)
        assert s > 0

    def test_small(self):
        """σ_numerical must be far below any macroscopic signal (< 1e-8)."""
        from quasim.ciir.multi_qubit.analysis.falsification import compute_sigma_numerical

        s = compute_sigma_numerical(50, 4, 0.01)
        assert s < 1e-8

    def test_scales_with_n_steps(self):
        from quasim.ciir.multi_qubit.analysis.falsification import compute_sigma_numerical

        s1 = compute_sigma_numerical(50, 4, 0.01)
        s2 = compute_sigma_numerical(100, 4, 0.01)
        assert s2 > s1

    def test_scales_with_dim(self):
        from quasim.ciir.multi_qubit.analysis.falsification import compute_sigma_numerical

        s4 = compute_sigma_numerical(50, 4, 0.01)
        s8 = compute_sigma_numerical(50, 8, 0.01)
        assert s8 > s4


# ===========================================================================
# run_ciir_trajectory
# ===========================================================================


class TestRunCIIRTrajectory:
    def test_returns_keys(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        for k in ("trajectory", "fidelity", "entropy", "model"):
            assert k in r

    def test_trajectory_length(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert len(r["trajectory"]) == N_STEPS + 1

    def test_fidelity_in_01(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        fid = r["fidelity"]
        assert np.all(fid >= -1e-9) and np.all(fid <= 1.0 + 1e-9)

    def test_density_matrices_hermitian(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        for rho in r["trajectory"]:
            assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_density_matrices_trace_one(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        for rho in r["trajectory"]:
            assert abs(np.trace(rho).real - 1.0) < 1e-9

    def test_model_label(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert r["model"] == "CIIR_A"

    def test_entropy_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_ciir_trajectory

        r = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert np.all(r["entropy"] >= -1e-10)


# ===========================================================================
# run_null_trajectory
# ===========================================================================


class TestRunNullTrajectory:
    def test_returns_keys(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_null_trajectory

        r = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        for k in ("trajectory", "fidelity", "entropy", "model"):
            assert k in r

    def test_model_label(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_null_trajectory

        r = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert r["model"] == "NULL_B"

    def test_fidelity_in_01(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_null_trajectory

        r = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert np.all(r["fidelity"] >= -1e-9) and np.all(r["fidelity"] <= 1.0 + 1e-9)

    def test_trace_preserved(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_null_trajectory

        r = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        for rho in r["trajectory"]:
            assert abs(np.trace(rho).real - 1.0) < 1e-9


# ===========================================================================
# Signal: CIIR > Null
# ===========================================================================


class TestSignalDirection:
    def test_ciir_fidelity_exceeds_null(self):
        """Model A (CIIR) fidelity should be ≥ Model B (null) at most time steps."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            run_ciir_trajectory,
            run_null_trajectory,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = c["fidelity"] - b["fidelity"]
        # At least half the time steps should show positive delta
        assert np.sum(delta > 0) >= len(delta) // 2

    def test_final_fidelity_ciir_gt_null(self):
        """At the final step, CIIR should outperform null."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            run_ciir_trajectory,
            run_null_trajectory,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        assert c["fidelity"][-1] >= b["fidelity"][-1] - 1e-8


# ===========================================================================
# compute_signal_metrics
# ===========================================================================


class TestSignalMetrics:
    def _run(self):
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        return compute_signal_metrics(c, b)

    def test_returns_expected_keys(self):
        m = self._run()
        for k in (
            "delta_O_max",
            "sigma_numerical",
            "snr",
            "fidelity_ciir",
            "fidelity_null",
            "delta_O",
            "entropy_ciir",
            "entropy_null",
        ):
            assert k in m

    def test_delta_O_max_positive(self):
        m = self._run()
        assert m["delta_O_max"] >= 0.0

    def test_snr_positive(self):
        m = self._run()
        assert m["snr"] > 0

    def test_delta_O_array_length(self):
        m = self._run()
        assert len(m["delta_O"]) == len(m["fidelity_ciir"])

    def test_snr_large(self):
        """SNR should be >> 5 (Verdict A territory) for this noise regime."""
        m = self._run()
        assert m["snr"] > 5.0


# ===========================================================================
# Artifact hypothesis tests
# ===========================================================================


class TestAH1:
    def test_returns_artifact_result(self):
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah1_bond_dimension,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah1_bond_dimension(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.label == "AH-1"
        assert isinstance(r.passed, bool)
        assert r.value >= 0

    def test_passes_for_small_N(self):
        """For exact density matrices (small N) the truncation artifact is tiny."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah1_bond_dimension,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah1_bond_dimension(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.passed


class TestAH2:
    def test_returns_artifact_result(self):
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah2_step_size,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah2_step_size(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.label == "AH-2"
        assert isinstance(r.passed, bool)

    def test_passes_for_stable_signal(self):
        """Signal should be stable under step-size halving (RK4 is order-4)."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah2_step_size,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah2_step_size(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.passed


class TestAH3:
    def test_returns_artifact_result(self):
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah3_initial_state,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah3_initial_state(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.label == "AH-3"
        assert isinstance(r.passed, bool)

    def test_passes_for_tiny_perturbation(self):
        """ε=1e-6 perturbation should leave ΔO_max stable."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah3_initial_state,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah3_initial_state(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta, epsilon=1e-6)
        assert r.passed


class TestAH4:
    def test_returns_artifact_result(self):
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah4_noise_sensitivity,
        )

        c = run_ciir_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        b = run_null_trajectory(N_SMALL, N_STEPS, DT, GAMMA, NOISE)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah4_noise_sensitivity(N_SMALL, N_STEPS, DT, GAMMA, NOISE, delta)
        assert r.label == "AH-4"
        assert isinstance(r.passed, bool)
        assert r.value >= 0


class TestAH5:
    def test_returns_artifact_result(self):
        from quasim.ciir.multi_qubit.analysis.falsification import test_ah5_n_scaling

        r = test_ah5_n_scaling(N_STEPS, DT, GAMMA, NOISE, N_list=[2, 4])
        assert r.label == "AH-5"
        assert isinstance(r.passed, bool)

    def test_passes_for_small_N_list(self):
        """Signal should be positive and bounded for N=2,4 — CIIR geometric prediction."""
        from quasim.ciir.multi_qubit.analysis.falsification import test_ah5_n_scaling

        r = test_ah5_n_scaling(N_STEPS, DT, GAMMA, NOISE, N_list=[2, 4])
        # CIIR predicts bounded positive signal across N (50% code subspace)
        assert r.passed

    def test_detail_contains_N_values(self):
        from quasim.ciir.multi_qubit.analysis.falsification import test_ah5_n_scaling

        r = test_ah5_n_scaling(N_STEPS, DT, GAMMA, NOISE, N_list=[2, 4])
        assert "N=" in r.detail or "N_list" in r.detail


# ===========================================================================
# classify_verdict
# ===========================================================================


class TestClassifyVerdict:
    def _fake_artifact(self, passed: bool, label: str = "AH-X"):
        from quasim.ciir.multi_qubit.analysis.falsification import ArtifactResult

        return ArtifactResult(
            label=label,
            passed=passed,
            value=0.001,
            threshold=0.01,
            detail="test",
            counterargument="none",
        )

    def test_verdict_A_high_snr_all_pass(self):
        from quasim.ciir.multi_qubit.analysis.falsification import classify_verdict

        arts = [self._fake_artifact(True, f"AH-{i}") for i in range(1, 6)]
        v = classify_verdict(delta_O_max=0.05, sigma_numerical=1e-13, artifact_results=arts)
        assert v.letter == "A"

    def test_verdict_C_low_snr(self):
        from quasim.ciir.multi_qubit.analysis.falsification import classify_verdict

        arts = [self._fake_artifact(True, f"AH-{i}") for i in range(1, 6)]
        # SNR < 1: delta_O_max < sigma
        v = classify_verdict(delta_O_max=1e-14, sigma_numerical=1e-13, artifact_results=arts)
        assert v.letter == "C"

    def test_verdict_B_medium_snr(self):
        from quasim.ciir.multi_qubit.analysis.falsification import classify_verdict

        arts = [self._fake_artifact(True, f"AH-{i}") for i in range(1, 6)]
        # SNR = 3 (between 1 and 5)
        v = classify_verdict(delta_O_max=3e-13, sigma_numerical=1e-13, artifact_results=arts)
        assert v.letter == "B"

    def test_verdict_C_artifact_fail(self):
        from quasim.ciir.multi_qubit.analysis.falsification import classify_verdict

        arts = [self._fake_artifact(True, f"AH-{i}") for i in range(1, 5)]
        arts.append(self._fake_artifact(False, "AH-5"))  # one failure
        # Any artifact failure → C regardless of SNR
        v = classify_verdict(delta_O_max=0.05, sigma_numerical=1e-13, artifact_results=arts)
        assert v.letter == "C"

    def test_verdict_fields(self):
        from quasim.ciir.multi_qubit.analysis.falsification import classify_verdict

        arts = [self._fake_artifact(True, f"AH-{i}") for i in range(1, 6)]
        v = classify_verdict(delta_O_max=0.05, sigma_numerical=1e-13, artifact_results=arts)
        assert v.snr > 0
        assert v.justification
        assert v.counterargument


# ===========================================================================
# run_falsification_protocol — end-to-end smoke test
# ===========================================================================


class TestFullProtocol:
    def _run(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_falsification_protocol

        return run_falsification_protocol(
            N=2,
            n_steps=15,
            dt=0.02,
            gamma=0.02,
            noise_strength=0.15,
            N_scan=[2, 4],
            run_secondary=False,
        )

    def test_returns_keys(self):
        r = self._run()
        for k in ("ciir", "null", "signal", "artifacts", "verdict", "params"):
            assert k in r

    def test_artifact_count(self):
        r = self._run()
        assert len(r["artifacts"]) == 5

    def test_verdict_is_A(self):
        """With noise > p* and all artifacts passing, the verdict should be A.

        SNR is >> 5σ (numerical floor is ~1e-14, signal is ~0.3).
        All artifact hypotheses should pass for N=2 (small exact-DM system).
        """
        r = self._run()
        # Signal is unambiguously strong; all AHs should pass for N=2
        assert r["verdict"].letter in (
            "A",
        ), f"Expected A but got {r['verdict'].letter}: {r['verdict'].justification}"

    def test_verdict_has_justification(self):
        r = self._run()
        assert r["verdict"].justification

    def test_all_artifacts_have_label(self):
        r = self._run()
        for label in ["AH-1", "AH-2", "AH-3", "AH-4", "AH-5"]:
            assert label in r["artifacts"]
            assert r["artifacts"][label].label == label


# ===========================================================================
# Secondary objectives (S1, S2, S3)
# ===========================================================================


class TestSecondaryS1:
    def test_returns_expected_keys(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s1_stress_regimes

        r = run_s1_stress_regimes(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        for k in (
            "delta_baseline",
            "delta_high_entanglement",
            "delta_long_memory",
            "delta_rapid_control",
        ):
            assert k in r

    def test_all_deltas_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s1_stress_regimes

        r = run_s1_stress_regimes(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        for k in (
            "delta_baseline",
            "delta_high_entanglement",
            "delta_long_memory",
            "delta_rapid_control",
        ):
            assert r[k] >= 0.0

    def test_trends_are_valid_strings(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s1_stress_regimes

        r = run_s1_stress_regimes(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        valid_trends = {"STRENGTHENED", "WEAKENED", "STABLE"}
        for k in ("trend_high_entanglement", "trend_long_memory", "trend_rapid_control"):
            assert r[k] in valid_trends


class TestSecondaryS2:
    def test_returns_expected_keys(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s2_controller_comparison

        r = run_s2_controller_comparison(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        for k in (
            "delta_lqr",
            "delta_pontryagin",
            "delta_natural_gradient",
            "geometric_enhances_signal",
            "nat_grad_vs_lqr_ratio",
        ):
            assert k in r

    def test_all_deltas_non_negative(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s2_controller_comparison

        r = run_s2_controller_comparison(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        assert r["delta_lqr"] >= 0.0
        assert r["delta_pontryagin"] >= 0.0
        assert r["delta_natural_gradient"] >= 0.0

    def test_geometric_enhances_is_bool(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s2_controller_comparison

        r = run_s2_controller_comparison(N=2, n_steps=15, dt=0.02, gamma=0.02, noise_strength=0.15)
        assert isinstance(r["geometric_enhances_signal"], (bool, np.bool_))


class TestSecondaryS3:
    def test_returns_expected_keys(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s3_alpha_consistency

        r = run_s3_alpha_consistency(N_list=[2, 4], gamma=0.02)
        for k in ("per_N", "n_star_diverge", "all_agree_5pct", "all_in_range"):
            assert k in r

    def test_per_N_structure(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s3_alpha_consistency

        r = run_s3_alpha_consistency(N_list=[2], gamma=0.02)
        entry = r["per_N"][0]
        for k in ("N", "alpha_spectral", "alpha_hausdorff", "agreement", "both_in_range"):
            assert k in entry

    def test_alpha_positive(self):
        from quasim.ciir.multi_qubit.analysis.falsification import run_s3_alpha_consistency

        r = run_s3_alpha_consistency(N_list=[2], gamma=0.02)
        for entry in r["per_N"]:
            assert entry["alpha_spectral"] > 0
            assert entry["alpha_hausdorff"] > 0


# ===========================================================================
# Epistemic constraint checks
# ===========================================================================


class TestEpistemicConstraints:
    def test_ec1_every_verdict_has_counterargument(self):
        """EC-1: every verdict must have a counterargument."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            ArtifactResult,
            classify_verdict,
        )

        arts = [ArtifactResult(f"AH-{i}", True, 0.001, 0.01, "d", "ca") for i in range(1, 6)]
        for delta, sigma in [(0.05, 1e-13), (3e-13, 1e-13), (1e-14, 1e-13)]:
            v = classify_verdict(delta, sigma, arts)
            assert v.counterargument, "EC-1 violated: missing counterargument"

    def test_ec3_sigma_reported(self):
        """EC-3: σ_numerical must be present in signal metrics."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
        )

        c = run_ciir_trajectory(2, 10, 0.02, 0.02, 0.15)
        b = run_null_trajectory(2, 10, 0.02, 0.02, 0.15)
        m = compute_signal_metrics(c, b)
        assert "sigma_numerical" in m
        assert m["sigma_numerical"] > 0

    def test_ec4_artifact_is_binary(self):
        """EC-4: artifact hypothesis results are strictly boolean."""
        from quasim.ciir.multi_qubit.analysis.falsification import (
            compute_signal_metrics,
            run_ciir_trajectory,
            run_null_trajectory,
            test_ah1_bond_dimension,
        )

        c = run_ciir_trajectory(2, 10, 0.02, 0.02, 0.15)
        b = run_null_trajectory(2, 10, 0.02, 0.02, 0.15)
        delta = compute_signal_metrics(c, b)["delta_O_max"]
        r = test_ah1_bond_dimension(2, 10, 0.02, 0.02, 0.15, delta)
        assert r.passed is True or r.passed is False

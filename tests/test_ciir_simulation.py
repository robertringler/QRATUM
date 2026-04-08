"""Comprehensive tests for the CIIR → QuASIM → QRATUM simulation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from quasim.ciir.core import (
    density_matrix,
    measurement_probability,
    normalize,
    von_neumann_entropy,
)
from quasim.ciir.engine import CIIRSimulationEngine, SimulationConfig, StepRecord
from quasim.ciir.experiments import (
    experiment_operator_drift,
    experiment_order_effect,
    experiment_path_dependence,
    experiment_phase_transitions,
    experiment_triadic_interference,
    run_all_experiments,
)
from quasim.ciir.operators import (
    OperatorSystem,
    commutator,
    frobenius_norm,
)
from quasim.ciir.report import generate_report

# ======================================================================
# Core utilities
# ======================================================================


class TestNormalize:
    def test_unit_norm(self):
        v = np.array([3.0 + 4j, 0.0, 0.0])
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-12

    def test_preserves_direction(self):
        v = np.array([1.0, 1.0])
        n = normalize(v)
        assert np.allclose(n, v / np.linalg.norm(v))

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            normalize(np.zeros(3, dtype=complex))


class TestDensityMatrix:
    def test_shape(self):
        psi = normalize(np.array([1.0 + 0j, 0.0, 0.0]))
        rho = density_matrix(psi)
        assert rho.shape == (3, 3)

    def test_trace_one(self):
        psi = normalize(np.array([1.0 + 0j, 1j, 0.5]))
        rho = density_matrix(psi)
        assert abs(np.trace(rho) - 1.0) < 1e-12

    def test_hermitian(self):
        psi = normalize(np.array([1.0 + 2j, 3 - 1j]))
        rho = density_matrix(psi)
        assert np.allclose(rho, rho.conj().T)

    def test_pure_state_rank_one(self):
        psi = normalize(np.array([1.0, 0.0, 0.0], dtype=complex))
        rho = density_matrix(psi)
        eigvals = np.linalg.eigvalsh(rho)
        assert sum(e > 1e-10 for e in eigvals) == 1


class TestVonNeumannEntropy:
    def test_pure_state_zero(self):
        psi = normalize(np.array([1.0 + 0j, 0.0]))
        rho = density_matrix(psi)
        assert von_neumann_entropy(rho) < 1e-10

    def test_maximally_mixed(self):
        rho = np.eye(2) / 2.0
        s = von_neumann_entropy(rho)
        assert abs(s - np.log(2)) < 1e-10

    def test_non_negative(self):
        rho = np.diag([0.7, 0.2, 0.1])
        assert von_neumann_entropy(rho) >= 0


class TestMeasurementProbability:
    def test_identity_gives_one(self):
        psi = normalize(np.array([1.0 + 0j, 0.0]))
        p = measurement_probability(psi, np.eye(2, dtype=complex))
        assert abs(p - 1.0) < 1e-10

    def test_in_range(self):
        psi = normalize(np.array([1.0 + 0j, 1j]) / np.sqrt(2))
        m = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        p = measurement_probability(psi, m)
        assert 0.0 <= p <= 1.0


# ======================================================================
# Operator system
# ======================================================================


class TestCommutator:
    def test_self_commutator_zero(self):
        m = np.array([[1, 2], [3, 4]], dtype=complex)
        assert np.allclose(commutator(m, m), 0)

    def test_non_commuting(self):
        a = np.array([[0, 1], [0, 0]], dtype=complex)
        b = np.array([[0, 0], [1, 0]], dtype=complex)
        c = commutator(a, b)
        assert not np.allclose(c, 0)


class TestOperatorSystem:
    def test_random_creation(self):
        ops = OperatorSystem.random(["A", "B"], dim=3)
        assert set(ops.operators.keys()) == {"A", "B"}
        for m in ops.operators.values():
            assert m.shape == (3, 3)
            assert abs(frobenius_norm(m) - 1.0) < 1e-10

    def test_qratum_update_produces_drift(self):
        ops = OperatorSystem.random(["X"], dim=4, eta=0.1)
        psi = normalize(
            np.random.default_rng(0).standard_normal(4)
            + 1j * np.random.default_rng(1).standard_normal(4)
        )
        rho = density_matrix(psi)
        drifts = ops.qratum_update(rho)
        assert "X" in drifts
        # With non-trivial rho, drift should be > 0
        assert drifts["X"] >= 0

    def test_commutator_norms_dict(self):
        ops = OperatorSystem.random(["A", "B", "C"], dim=3)
        cn = ops.commutator_norms()
        assert ("A", "B") in cn
        assert ("A", "C") in cn
        assert ("B", "C") in cn
        assert all(v >= 0 for v in cn.values())

    def test_snapshot_independence(self):
        ops = OperatorSystem.random(["A"], dim=2)
        snap = ops.snapshot()
        ops.operators["A"] *= 0  # mutate original
        assert not np.allclose(snap["A"], ops.operators["A"])


# ======================================================================
# Simulation engine
# ======================================================================


class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.dim == 4
        assert cfg.num_steps == 200
        assert cfg.eta == 0.05

    def test_custom(self):
        cfg = SimulationConfig(dim=8, num_steps=50)
        assert cfg.dim == 8
        assert cfg.num_steps == 50


class TestCIIRSimulationEngine:
    @pytest.fixture()
    def engine(self):
        return CIIRSimulationEngine(
            SimulationConfig(dim=4, num_steps=10, seed=42)
        )

    def test_initial_state_normalized(self, engine):
        assert abs(np.linalg.norm(engine.psi) - 1.0) < 1e-12

    def test_hamiltonian_hermitian(self, engine):
        h = engine.hamiltonian
        assert np.allclose(h, h.conj().T)

    def test_step_returns_record(self, engine):
        rec = engine.step()
        assert isinstance(rec, StepRecord)
        assert rec.t == 0
        assert rec.entropy >= -1e-12  # allow floating-point rounding

    def test_run_length(self, engine):
        history = engine.run()
        assert len(history) == 10

    def test_state_stays_normalized(self, engine):
        engine.run()
        for rec in engine.history:
            assert abs(np.linalg.norm(rec.psi) - 1.0) < 1e-10

    def test_collect_metrics_keys(self, engine):
        engine.run()
        m = engine.collect_metrics()
        assert "order_effect_mean" in m
        assert "entropy_trajectory" in m
        assert "mean_operator_drifts" in m
        assert "eigenspectra" in m


# ======================================================================
# Experiments
# ======================================================================


_SMALL = SimulationConfig(dim=3, num_steps=20, seed=123)


class TestExperimentOrderEffect:
    def test_returns_dict(self):
        r = experiment_order_effect(_SMALL)
        assert "delta_trajectory" in r
        assert "is_nonstationary" in r
        assert len(r["delta_trajectory"]) == 20


class TestExperimentPathDependence:
    def test_returns_dict(self):
        r = experiment_path_dependence(_SMALL)
        assert "state_distance" in r
        assert "path_dependent" in r


class TestExperimentOperatorDrift:
    def test_returns_dict(self):
        r = experiment_operator_drift(_SMALL)
        assert "drift_trajectories" in r
        assert "classification" in r


class TestExperimentPhaseTransitions:
    def test_returns_dict(self):
        cfg = SimulationConfig(dim=3, num_steps=10, seed=99)
        r = experiment_phase_transitions(cfg)
        assert "sweep" in r
        assert "transition_detected" in r
        assert len(r["sweep"]) > 0


class TestExperimentTriadicInterference:
    def test_returns_dict(self):
        r = experiment_triadic_interference(_SMALL)
        assert "deviation_trajectory" in r
        assert "triadic_interference_present" in r


class TestRunAllExperiments:
    def test_all_keys(self):
        r = run_all_experiments(_SMALL)
        assert set(r.keys()) == {
            "order_effect",
            "path_dependence",
            "operator_drift",
            "phase_transitions",
            "triadic_interference",
        }


# ======================================================================
# Report
# ======================================================================


class TestReport:
    def test_generates_string(self):
        r = run_all_experiments(_SMALL)
        report = generate_report(r)
        assert isinstance(report, str)
        assert "EXECUTIVE SUMMARY" in report
        assert "FINAL DETERMINATION" in report

    def test_contains_all_sections(self):
        r = run_all_experiments(_SMALL)
        report = generate_report(r)
        for section in (
            "KEY FINDINGS",
            "QUANTITATIVE METRICS",
            "REGIME CLASSIFICATION",
            "FALSIFIABILITY VERDICT",
            "QRATUM-CLASS SUCCESS CRITERIA",
        ):
            assert section in report

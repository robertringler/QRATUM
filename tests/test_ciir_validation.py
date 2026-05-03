"""Tests for the CIIR multi-scenario validation runner."""

from __future__ import annotations

import json
import os
import tempfile

from quasim.ciir.simulation.engine import SimulationConfig
from quasim.ciir.validation_runner import (
    ScenarioMetrics,
    ScenarioResult,
    ValidationReport,
    _config_to_dict,
    _run_scenario,
    generate_convergence_plots,
    run_all_validations,
    run_baseline,
    run_constraint_sweep,
    run_high_dimensional,
    run_observer_sweep,
    run_step_sweep,
    run_stress_test,
)

# ================================================================
# ScenarioMetrics
# ================================================================


class TestScenarioMetrics:
    def test_defaults(self):
        m = ScenarioMetrics()
        assert m.steps == []
        assert m.losses == []
        assert m.gradient_norms == []

    def test_append(self):
        m = ScenarioMetrics()
        m.steps.append(1)
        m.losses.append(0.5)
        assert len(m.steps) == 1
        assert m.losses[0] == 0.5


# ================================================================
# ScenarioResult
# ================================================================


class TestScenarioResult:
    def test_to_dict(self):
        metrics = ScenarioMetrics(steps=[1, 2], losses=[1.0, 0.5])
        r = ScenarioResult(
            name="test",
            config={"rank": 4},
            metrics=metrics,
            summary={"final_loss": 0.5},
            elapsed_s=1.5,
            passed=True,
            anomalies=[],
        )
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is True
        assert d["summary"]["final_loss"] == 0.5
        assert d["metrics"]["losses"] == [1.0, 0.5]

    def test_anomalies(self):
        r = ScenarioResult(
            name="test",
            config={},
            metrics=ScenarioMetrics(),
            anomalies=["NaN detected"],
            passed=False,
        )
        assert not r.passed
        assert "NaN detected" in r.anomalies


# ================================================================
# ValidationReport
# ================================================================


class TestValidationReport:
    def test_empty(self):
        r = ValidationReport()
        assert r.n_total == 0
        assert r.n_passed == 0

    def test_counts(self):
        r = ValidationReport(
            scenarios=[
                ScenarioResult(name="a", config={}, metrics=ScenarioMetrics(), passed=True),
                ScenarioResult(name="b", config={}, metrics=ScenarioMetrics(), passed=False),
                ScenarioResult(name="c", config={}, metrics=ScenarioMetrics(), passed=True),
            ]
        )
        assert r.n_total == 3
        assert r.n_passed == 2

    def test_to_dict(self):
        r = ValidationReport(
            scenarios=[
                ScenarioResult(name="a", config={}, metrics=ScenarioMetrics(), passed=True),
            ],
            total_elapsed_s=5.0,
        )
        d = r.to_dict()
        assert d["summary"]["total_scenarios"] == 1
        assert d["summary"]["passed"] == 1
        assert d["summary"]["total_elapsed_s"] == 5.0

    def test_save_json(self):
        r = ValidationReport(
            scenarios=[
                ScenarioResult(name="a", config={}, metrics=ScenarioMetrics(), passed=True),
            ],
            total_elapsed_s=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            r.save_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["total_scenarios"] == 1


# ================================================================
# _config_to_dict
# ================================================================


class TestConfigToDict:
    def test_serialization(self):
        config = SimulationConfig(rank=4, rep_dim=8, n_constraints=3)
        d = _config_to_dict(config)
        assert d["rank"] == 4
        assert d["rep_dim"] == 8
        assert d["n_constraints"] == 3
        assert "seed" in d


# ================================================================
# _run_scenario
# ================================================================


class TestRunScenario:
    def test_basic(self):
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=2,
            n_constraints=2,
            n_observers=2,
            n_steps=10,
            learning_rate=0.01,
            seed=42,
            output_dir=os.path.join(tempfile.gettempdir(), "ciir_validation_test", "basic"),
            log_metrics=False,
            log_tensors=False,
        )
        result = _run_scenario("test_basic", config, verbose=False)
        assert result.name == "test_basic"
        assert result.passed
        assert len(result.metrics.steps) == 10
        assert len(result.metrics.losses) == 10
        assert result.summary["n_steps"] == 10

    def test_summary_has_required_keys(self):
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=2,
            n_constraints=2,
            n_observers=2,
            n_steps=10,
            seed=42,
            output_dir=os.path.join(tempfile.gettempdir(), "ciir_validation_test", "summary"),
            log_metrics=False,
            log_tensors=False,
        )
        result = _run_scenario("test_summary", config, verbose=False)
        summary = result.summary
        assert "initial_loss" in summary
        assert "final_loss" in summary
        assert "mean_gradient_norm" in summary
        assert "mean_purity" in summary
        assert "mean_entropy" in summary
        assert "throughput_steps_per_s" in summary
        assert "non_commutative" in summary


# ================================================================
# Individual scenarios (short runs to verify they work)
# ================================================================


class TestScenarios:
    def test_baseline(self):
        r = run_baseline(verbose=False)
        assert r.passed
        assert r.name == "baseline"
        assert len(r.metrics.steps) > 0

    def test_constraint_sweep(self):
        results = run_constraint_sweep(verbose=False)
        assert len(results) == 5  # 1, 3, 5, 7, 10
        for r in results:
            assert "constraint_sweep" in r.name
            assert len(r.metrics.steps) > 0

    def test_observer_sweep(self):
        results = run_observer_sweep(verbose=False)
        assert len(results) == 4  # 1, 2, 3, 5
        for r in results:
            assert "observer_sweep" in r.name

    def test_step_sweep(self):
        results = run_step_sweep(verbose=False)
        assert len(results) == 4  # 50, 100, 500, 1000
        # Verify step counts match
        expected_steps = [50, 100, 500, 1000]
        for r, expected in zip(results, expected_steps):
            assert len(r.metrics.steps) == expected

    def test_high_dimensional(self):
        r = run_high_dimensional(verbose=False)
        assert r.passed
        assert r.config["rep_dim"] == 16
        assert r.config["rank"] == 8

    def test_stress_test(self):
        r = run_stress_test(verbose=False)
        assert r.passed
        assert r.config["batch_size"] == 32


# ================================================================
# Full validation suite
# ================================================================


class TestFullValidation:
    def test_run_all(self):
        report = run_all_validations(verbose=False)
        assert report.n_total == 16  # 1 + 5 + 4 + 4 + 1 + 1
        assert report.n_passed > 0
        assert report.total_elapsed_s > 0

    def test_save_json(self):
        report = run_all_validations(verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            report.save_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["total_scenarios"] == 16


# ================================================================
# Convergence plots
# ================================================================


class TestConvergencePlots:
    def test_generate_plots(self):
        """Verify plots are generated without error."""
        import matplotlib

        matplotlib.use("Agg")

        # Run a tiny subset for speed
        config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=2,
            n_constraints=2,
            n_observers=2,
            n_steps=10,
            seed=42,
            output_dir=os.path.join(tempfile.gettempdir(), "ciir_validation_test", "plots"),
            log_metrics=False,
            log_tensors=False,
        )
        r1 = _run_scenario("plot_test_a", config, verbose=False)
        config.seed = 99
        r2 = _run_scenario("plot_test_b", config, verbose=False)

        report = ValidationReport(scenarios=[r1, r2], total_elapsed_s=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_convergence_plots(report, output_dir=tmpdir)
            assert len(paths) == 5
            for p in paths:
                assert os.path.exists(p)
                assert p.endswith(".png")

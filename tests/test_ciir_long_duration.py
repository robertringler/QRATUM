"""Tests for the CIIR long-duration simulation runner."""

from __future__ import annotations

import json
import os
import tempfile

from quasim.ciir.long_duration_runner import (LongRunConfig, LongRunResult,
                                              StepRecord, _detect_anomaly,
                                              _recover_engine,
                                              generate_long_run_plots,
                                              run_long_duration)
from quasim.ciir.simulation.engine import (CIIRSimulationEngine,
                                           SimulationConfig)

# ================================================================
# StepRecord
# ================================================================


class TestStepRecord:
    def test_to_dict(self):
        r = StepRecord(
            step=10,
            timestamp_s=1.5,
            loss=0.5,
            gradient_norm=0.01,
            purity=0.9,
            entropy=0.1,
            constraint_loss=0.2,
            observer_loss=0.1,
            entropy_loss=0.05,
            step_time_ms=0.8,
            max_commutator_norm=0.3,
        )
        d = r.to_dict()
        assert d["step"] == 10
        assert d["loss"] == 0.5
        assert d["max_commutator_norm"] == 0.3

    def test_defaults(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=1.0,
            gradient_norm=0.1,
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert r.max_commutator_norm == 0.0


# ================================================================
# LongRunResult
# ================================================================


class TestLongRunResult:
    def test_empty(self):
        r = LongRunResult()
        assert r.n_steps == 0
        s = r.summary()
        assert s["n_steps"] == 0

    def test_with_records(self):
        records = [
            StepRecord(
                step=i,
                timestamp_s=i * 0.01,
                loss=1.0 - i * 0.01,
                gradient_norm=0.1,
                purity=0.5,
                entropy=0.5,
                constraint_loss=0.3,
                observer_loss=0.2,
                entropy_loss=0.1,
                step_time_ms=1.0,
            )
            for i in range(50)
        ]
        r = LongRunResult(records=records, total_elapsed_s=5.0, config={"seed": 42})
        assert r.n_steps == 50
        assert r.final_loss == records[-1].loss
        s = r.summary()
        assert s["n_steps"] == 50
        assert s["total_elapsed_s"] == 5.0
        assert s["throughput_steps_per_s"] > 0

    def test_save_json(self):
        r = LongRunResult(
            records=[
                StepRecord(
                    step=1,
                    timestamp_s=0.0,
                    loss=1.0,
                    gradient_norm=0.1,
                    purity=0.5,
                    entropy=0.5,
                    constraint_loss=0.3,
                    observer_loss=0.2,
                    entropy_loss=0.1,
                    step_time_ms=1.0,
                )
            ],
            total_elapsed_s=1.0,
            config={"seed": 42},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.json")
            r.save_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["n_steps"] == 1

    def test_save_checkpoint(self):
        records = [
            StepRecord(
                step=i,
                timestamp_s=i * 0.01,
                loss=1.0,
                gradient_norm=0.1,
                purity=0.5,
                entropy=0.5,
                constraint_loss=0.3,
                observer_loss=0.2,
                entropy_loss=0.1,
                step_time_ms=1.0,
            )
            for i in range(200)
        ]
        r = LongRunResult(records=records, total_elapsed_s=2.0, config={"seed": 42})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.json")
            r.save_checkpoint(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["n_total_records"] == 200
            # Checkpoint only keeps last 100
            assert len(data["recent_records"]) == 100

    def test_to_dict(self):
        r = LongRunResult(config={"seed": 42}, total_elapsed_s=1.0)
        d = r.to_dict()
        assert "config" in d
        assert "summary" in d
        assert "records" in d


# ================================================================
# Anomaly detection
# ================================================================


class TestAnomalyDetection:
    def test_no_anomaly(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=1.0,
            gradient_norm=0.1,
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is None

    def test_nan_loss(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=float("nan"),
            gradient_norm=0.1,
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is not None
        assert "NaN" in _detect_anomaly(r)

    def test_inf_loss(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=float("inf"),
            gradient_norm=0.1,
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is not None
        assert "Inf" in _detect_anomaly(r)

    def test_gradient_explosion(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=1.0,
            gradient_norm=1e11,
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is not None
        assert "explosion" in _detect_anomaly(r).lower()

    def test_nan_gradient(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=1.0,
            gradient_norm=float("nan"),
            purity=0.5,
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is not None
        assert "NaN gradient" in _detect_anomaly(r)

    def test_nan_purity(self):
        r = StepRecord(
            step=1,
            timestamp_s=0.0,
            loss=1.0,
            gradient_norm=0.1,
            purity=float("nan"),
            entropy=0.5,
            constraint_loss=0.3,
            observer_loss=0.2,
            entropy_loss=0.1,
            step_time_ms=1.0,
        )
        assert _detect_anomaly(r) is not None
        assert "purity" in _detect_anomaly(r).lower()


# ================================================================
# Recovery
# ================================================================


class TestRecovery:
    def test_recover_engine(self):
        cfg = LongRunConfig(seed=42)
        sim_config = SimulationConfig(
            rank=4,
            rep_dim=8,
            batch_size=2,
            n_constraints=2,
            n_observers=2,
            n_steps=0,
            seed=42,
            output_dir=os.path.join(tempfile.gettempdir(), "ciir_recovery_test"),
            log_metrics=False,
            log_tensors=False,
        )
        engine = CIIRSimulationEngine(sim_config)
        engine.init()
        engine.current_step = 100

        recovered = _recover_engine(engine, cfg, attempt=1)
        assert recovered.running
        assert recovered.current_step == 100
        assert recovered.ciir_state is not None


# ================================================================
# Short-duration run
# ================================================================


class TestRunLongDuration:
    def test_step_limited(self):
        """Run a short step-limited simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                duration_hours=0,
                max_steps=50,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            assert result.n_steps == 50
            assert result.final_loss < float("inf")
            assert not result.aborted
            assert result.total_elapsed_s > 0
            # Check final_results.json was saved
            assert os.path.exists(os.path.join(tmpdir, "final_results.json"))

    def test_time_limited(self):
        """Run a very short time-limited simulation (0.001 hours = 3.6s)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                duration_hours=0.001,  # ~3.6 seconds
                max_steps=0,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            assert result.n_steps > 0
            assert not result.aborted

    def test_custom_config(self):
        """Run with custom LongRunConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = LongRunConfig(
                duration_hours=0,
                max_steps=20,
                rank=4,
                rep_dim=8,
                batch_size=2,
                n_constraints=2,
                n_observers=2,
                seed=99,
                checkpoint_interval=10,
                log_interval=5,
                output_dir=tmpdir,
            )
            result = run_long_duration(config=cfg, verbose=False)
            assert result.n_steps == 20
            assert result.config["seed"] == 99

    def test_checkpoint_saved(self):
        """Verify checkpoints are saved at configured intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                config=LongRunConfig(
                    duration_hours=0,
                    max_steps=25,
                    checkpoint_interval=10,
                    seed=42,
                    output_dir=tmpdir,
                ),
                verbose=False,
            )
            assert result.n_steps == 25
            # Checkpoint at step 10 and 20
            assert os.path.exists(os.path.join(tmpdir, "checkpoint_step10.json"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoint_step20.json"))

    def test_summary_keys(self):
        """Verify summary contains expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                duration_hours=0,
                max_steps=10,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            s = result.summary()
            assert "n_steps" in s
            assert "initial_loss" in s
            assert "final_loss" in s
            assert "throughput_steps_per_s" in s
            assert "mean_purity" in s
            assert "mean_entropy" in s

    def test_commutator_norm_recorded(self):
        """Verify observer non-commutativity is recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                duration_hours=0,
                max_steps=5,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            # With 2 observers, commutator norm should be recorded
            assert result.records[0].max_commutator_norm >= 0


# ================================================================
# Plots
# ================================================================


class TestLongRunPlots:
    def test_generate_plots(self):
        """Verify plots are generated without error."""
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_long_duration(
                duration_hours=0,
                max_steps=20,
                seed=42,
                output_dir=tmpdir,
                verbose=False,
            )
            paths = generate_long_run_plots(result, output_dir=tmpdir)
            assert len(paths) == 6
            for p in paths:
                assert os.path.exists(p)
                assert p.endswith(".png")

    def test_empty_result_no_plots(self):
        """Verify empty result produces no plots."""
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = LongRunResult(config={"output_dir": tmpdir})
            paths = generate_long_run_plots(result, output_dir=tmpdir)
            assert len(paths) == 0

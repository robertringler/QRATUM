"""Tests for CIIR → QuASIM → QRATUM simulation package.

Tests cover:
    - CIIR Module: CIIRState, CIIRObserver, CIIRVisualizer
    - QuASIM Module: QuASIMTensor, TensorRuntime, TensorVisualizer
    - QRATUM Module: QRATUMHardware, PerformanceDashboard
    - Export: ScreenshotCapture, TensorLogger, MetricsCSVWriter
    - Dashboard: MetricsDashboard, MetricsSnapshot
    - Interactive: InteractiveController, ParameterPreset
    - Engine: CIIRSimulationEngine, run_and_capture_simulation
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasim.ciir.loss import CIIRLoss
from quasim.ciir.theory import build_default_theory

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def theory():
    """Default CIIR theory for testing."""
    return build_default_theory(rank=4, rep_dim=4, n_constraints=2, n_observers=2, seed=42)


@pytest.fixture
def rho_batch(theory):
    """Batch of density matrices."""
    rng = np.random.default_rng(42)
    return theory.manifold.density_matrix(
        theory.manifold.random_state(batch=2, rng=rng)
    )


@pytest.fixture
def loss_fn(theory):
    """CIIR loss function."""
    return CIIRLoss(theory=theory, entropy_weight=0.01)


@pytest.fixture
def tmpdir():
    """Temporary directory for test output."""
    with tempfile.TemporaryDirectory(prefix="ciir_test_") as d:
        yield d


# ================================================================
# CIIR Module Tests
# ================================================================


class TestCIIRState:
    """Tests for CIIRState."""

    def test_basic_properties(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory, step=0)
        assert state.batch_size == 2
        assert state.dim == 4
        assert state.step == 0

    def test_eigenvalues(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        eigs = state.eigenvalues
        assert eigs.shape == (2, 4)
        # Eigenvalues of density matrix should be non-negative
        assert np.all(eigs >= -1e-10)
        # Should sum to ~1 (trace = 1)
        np.testing.assert_allclose(eigs.sum(axis=1), 1.0, atol=1e-6)

    def test_purity(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        purity = state.purity
        assert purity.shape == (2,)
        # Purity in [1/D, 1]
        assert np.all(purity >= 1.0 / 4.0 - 1e-6)
        assert np.all(purity <= 1.0 + 1e-6)

    def test_entropy(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        entropy = state.entropy
        assert entropy.shape == (2,)
        # Entropy should be non-negative
        assert np.all(entropy >= -1e-6)
        # Max entropy = log(D)
        assert np.all(entropy <= np.log(4) + 1e-6)

    def test_manifold_embedding_3d(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        coords = state.manifold_embedding_3d()
        assert coords.shape == (2, 3)

    def test_constraint_values(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        cv = state.constraint_values
        assert cv.shape == (2, 2)

    def test_cache_invalidation(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        _ = state.purity
        state.invalidate_cache()
        assert state._purity is None

    def test_constraint_surface_samples(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        points, violations = state.constraint_surface_samples(n_samples=20)
        assert points.shape == (20, 3)
        assert violations.shape == (20,)

    def test_to_dict(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRState

        state = CIIRState(rho=rho_batch, theory=theory, step=5)
        d = state.to_dict()
        assert d["step"] == 5
        assert d["batch_size"] == 2
        assert d["dim"] == 4
        assert "purity" in d


class TestCIIRObserver:
    """Tests for CIIRObserver."""

    def test_observe(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRObserver, CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        observer = CIIRObserver(theory=theory)
        result = observer.observe(state, observer_index=0)
        assert "probabilities" in result
        assert "outcome" in result
        assert result["pre_entropy"] >= 0

    def test_commutator_matrix(self, theory):
        from quasim.ciir.simulation.ciir_module import CIIRObserver

        observer = CIIRObserver(theory=theory)
        comm = observer.commutator_matrix()
        assert comm.shape == (2, 2)
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(comm), 0.0, atol=1e-10)
        # Should be symmetric
        np.testing.assert_allclose(comm, comm.T, atol=1e-10)

    def test_expectation_values(self, theory, rho_batch):
        from quasim.ciir.simulation.ciir_module import CIIRObserver, CIIRState

        state = CIIRState(rho=rho_batch, theory=theory)
        observer = CIIRObserver(theory=theory)
        exps = observer.expectation_values(state)
        assert exps.shape == (2, 2)


# ================================================================
# QuASIM Module Tests
# ================================================================


class TestQuASIMTensor:
    """Tests for QuASIMTensor."""

    def test_creation(self):
        from quasim.ciir.simulation.quasim_module import QuASIMTensor

        t = QuASIMTensor(
            data=np.eye(4, dtype=np.float64),
            name="identity",
            dims=("rep", "rep"),
        )
        assert t.shape == (4, 4)
        assert t.frobenius_norm == pytest.approx(2.0, abs=1e-6)

    def test_spectral_norm(self):
        from quasim.ciir.simulation.quasim_module import QuASIMTensor

        t = QuASIMTensor(data=2.0 * np.eye(3))
        assert t.spectral_norm == pytest.approx(2.0, abs=1e-6)

    def test_trace(self):
        from quasim.ciir.simulation.quasim_module import QuASIMTensor

        t = QuASIMTensor(data=np.eye(5))
        assert t.trace() == pytest.approx(5.0, abs=1e-6)

    def test_spectrum(self):
        from quasim.ciir.simulation.quasim_module import QuASIMTensor

        m = np.diag([1.0, 2.0, 3.0])
        t = QuASIMTensor(data=m)
        spec = t.spectrum()
        np.testing.assert_allclose(sorted(spec), [1.0, 2.0, 3.0], atol=1e-6)

    def test_to_dict(self):
        from quasim.ciir.simulation.quasim_module import QuASIMTensor

        t = QuASIMTensor(data=np.eye(3), name="test")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["shape"] == [3, 3]


class TestTensorRuntime:
    """Tests for TensorRuntime."""

    def test_compute_loss_and_gradient(self, theory, loss_fn, rho_batch):
        from quasim.ciir.simulation.quasim_module import TensorRuntime

        rt = TensorRuntime(theory=theory, loss_fn=loss_fn)
        loss, grad, components = rt.compute_loss_and_gradient(rho_batch)
        assert isinstance(loss, float)
        assert grad.shape == rho_batch.shape
        assert "constraint" in components
        assert "observer" in components

    def test_gradient_step(self, theory, loss_fn, rho_batch):
        from quasim.ciir.simulation.quasim_module import TensorRuntime

        rt = TensorRuntime(theory=theory, loss_fn=loss_fn)
        rho_new, loss = rt.gradient_step(rho_batch, lr=0.01)
        assert rho_new.shape == rho_batch.shape

    def test_contraction_log(self, theory, loss_fn, rho_batch):
        from quasim.ciir.simulation.quasim_module import TensorRuntime

        rt = TensorRuntime(theory=theory, loss_fn=loss_fn)
        rt.compute_loss_and_gradient(rho_batch)
        assert len(rt.contraction_log) == 1
        assert "loss" in rt.contraction_log[0]

    def test_constraint_eval(self, theory, loss_fn, rho_batch):
        from quasim.ciir.simulation.quasim_module import TensorRuntime

        rt = TensorRuntime(theory=theory, loss_fn=loss_fn)
        cv = rt.constraint_eval(rho_batch)
        assert cv.shape == (2, 2)

    def test_observer_eval(self, theory, loss_fn, rho_batch):
        from quasim.ciir.simulation.quasim_module import TensorRuntime

        rt = TensorRuntime(theory=theory, loss_fn=loss_fn)
        ov = rt.observer_eval(rho_batch)
        assert ov.shape == (2, 2)


# ================================================================
# QRATUM Module Tests
# ================================================================


class TestQRATUMHardware:
    """Tests for QRATUMHardware."""

    def test_inject_constraints(self, theory, rho_batch):
        from quasim.ciir.config import CIIRConfig
        from quasim.ciir.simulation.qratum_module import QRATUMHardware

        hw = QRATUMHardware(theory=theory, config=CIIRConfig(seed=42))
        result = hw.inject_constraints(rho_batch)
        assert result.shape == rho_batch.shape
        # Should still be valid density matrices
        for b in range(result.shape[0]):
            np.testing.assert_allclose(np.trace(result[b]), 1.0, atol=1e-4)

    def test_record_and_summary(self, theory):
        from quasim.ciir.config import CIIRConfig
        from quasim.ciir.simulation.qratum_module import QRATUMHardware

        hw = QRATUMHardware(theory=theory, config=CIIRConfig(seed=42))
        hw.record_step_metrics(0, 1.5, 0.5, np.array([0.1, 0.2]), 0.001)
        hw.record_step_metrics(1, 1.2, 0.3, np.array([0.05, 0.1]), 0.001)
        assert len(hw.metrics_log) == 2
        summary = hw.device_summary()
        assert summary["performance"]["total_steps"] == 2


class TestPerformanceDashboard:
    """Tests for PerformanceDashboard."""

    def test_plot_dashboard(self, theory, tmpdir):
        from quasim.ciir.config import CIIRConfig
        from quasim.ciir.simulation.qratum_module import (PerformanceDashboard,
                                                          QRATUMHardware)

        hw = QRATUMHardware(theory=theory, config=CIIRConfig(seed=42))
        for i in range(10):
            hw.record_step_metrics(i, 1.0 - i * 0.05, 0.5 - i * 0.02,
                                   np.array([0.1, 0.05]), 0.001)

        dash = PerformanceDashboard()
        path = os.path.join(tmpdir, "dashboard.png")
        fig = dash.plot_full_dashboard(hw, save_path=path)
        assert fig is not None
        assert os.path.exists(path)


# ================================================================
# Export Tests
# ================================================================


class TestExport:
    """Tests for export utilities."""

    def test_screenshot_capture(self, tmpdir):
        import matplotlib.pyplot as plt

        from quasim.ciir.simulation.export import ScreenshotCapture

        cap = ScreenshotCapture(os.path.join(tmpdir, "screenshots"))
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = cap.capture(fig, step=0)
        plt.close(fig)
        assert os.path.exists(path)
        assert cap.count == 1

    def test_tensor_logger(self, tmpdir):
        from quasim.ciir.simulation.export import TensorLogger

        path = os.path.join(tmpdir, "tensors.json")
        logger = TensorLogger(path)
        logger.log(0, {"loss": 1.5, "norms": [0.1, 0.2]})
        logger.log(1, {"loss": 1.2, "norms": [0.05, 0.1]})
        logger.flush()
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["step"] == 0

    def test_metrics_csv_writer(self, tmpdir):
        from quasim.ciir.simulation.export import MetricsCSVWriter

        path = os.path.join(tmpdir, "metrics.csv")
        with MetricsCSVWriter(path, ["step", "loss"]) as writer:
            writer.write_row({"step": 0, "loss": 1.5})
            writer.write_row({"step": 1, "loss": 1.2})
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows


# ================================================================
# Dashboard Tests
# ================================================================


class TestMetricsDashboard:
    """Tests for MetricsDashboard."""

    def test_record_and_converge(self):
        from quasim.ciir.simulation.dashboard import (MetricsDashboard,
                                                      MetricsSnapshot)

        db = MetricsDashboard()
        for i in range(20):
            db.record(MetricsSnapshot(step=i, total_loss=1.0, gradient_norm=0.1))
        assert len(db.history) == 20
        assert db.is_converged(tol=0.01)

    def test_not_converged(self):
        from quasim.ciir.simulation.dashboard import (MetricsDashboard,
                                                      MetricsSnapshot)

        db = MetricsDashboard()
        for i in range(20):
            db.record(MetricsSnapshot(step=i, total_loss=float(i)))
        assert not db.is_converged(tol=0.01)

    def test_plot(self, tmpdir):
        from quasim.ciir.simulation.dashboard import (MetricsDashboard,
                                                      MetricsSnapshot)

        db = MetricsDashboard()
        for i in range(30):
            db.record(MetricsSnapshot(
                step=i, total_loss=1.0 / (i + 1), gradient_norm=0.5 / (i + 1),
                purity=0.3, entropy=1.2,
                constraint_violations=[0.1, 0.05],
            ))
        path = os.path.join(tmpdir, "dashboard.png")
        fig = db.plot(save_path=path)
        assert fig is not None
        assert os.path.exists(path)


# ================================================================
# Interactive Tests
# ================================================================


class TestInteractiveController:
    """Tests for InteractiveController."""

    def test_parameter_setting(self):
        from quasim.ciir.simulation.interactive import InteractiveController

        ctrl = InteractiveController(n_constraints=3, n_observers=2)
        ctrl.set_learning_rate(0.05)
        assert ctrl.learning_rate == 0.05
        ctrl.set_constraint_weight(1, 2.0)
        assert ctrl.constraint_weights[1] == 2.0

    def test_preset(self):
        from quasim.ciir.simulation.interactive import (InteractiveController,
                                                        ParameterPreset)

        ctrl = InteractiveController(n_constraints=3, n_observers=2)
        preset = ParameterPreset(name="fast", learning_rate=0.1, entropy_weight=0.001)
        ctrl.apply_preset(preset)
        assert ctrl.learning_rate == 0.1
        assert ctrl.entropy_weight == 0.001

    def test_callback(self):
        from quasim.ciir.simulation.interactive import InteractiveController

        ctrl = InteractiveController(n_constraints=2, n_observers=1)
        changes = []
        ctrl.on_change(lambda param, val: changes.append((param, val)))
        ctrl.set_learning_rate(0.02)
        assert len(changes) == 1
        assert changes[0] == ("learning_rate", 0.02)


# ================================================================
# Engine Integration Test
# ================================================================


class TestSimulationEngine:
    """Integration tests for CIIRSimulationEngine."""

    def test_run_and_capture(self, tmpdir):
        from quasim.ciir.simulation.engine import (SimulationConfig,
                                                   run_and_capture_simulation)

        engine = run_and_capture_simulation(
            config=SimulationConfig(
                rep_dim=4,
                batch_size=2,
                n_constraints=2,
                n_observers=1,
                n_steps=10,
                screenshot_interval=5,
                output_dir=tmpdir,
            ),
        )

        # Check engine state
        assert engine.current_step == 10
        assert len(engine.dashboard.history) == 10

        # Check output files exist
        assert os.path.exists(os.path.join(tmpdir, "metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "tensor_log.json"))
        assert os.path.exists(os.path.join(tmpdir, "dashboard.png"))

        # Check screenshots
        screenshots = os.listdir(os.path.join(tmpdir, "screenshots"))
        assert len(screenshots) == 2  # steps 5 and 10

    def test_step_by_step(self):
        from quasim.ciir.simulation.engine import (CIIRSimulationEngine,
                                                   SimulationConfig)

        cfg = SimulationConfig(
            rep_dim=4, batch_size=2, n_constraints=2, n_observers=1,
            n_steps=5, output_dir="/tmp/ciir_step_test",
        )
        engine = CIIRSimulationEngine(cfg)
        engine.init()

        for _ in range(5):
            snapshot = engine.step()
            assert snapshot.step > 0
            assert isinstance(snapshot.total_loss, float)
            assert snapshot.gradient_norm >= 0

        engine.shutdown()

    def test_convergence_check(self):
        from quasim.ciir.simulation.dashboard import (MetricsDashboard,
                                                      MetricsSnapshot)

        db = MetricsDashboard()
        # Feed identical losses → should converge
        for i in range(20):
            db.record(MetricsSnapshot(step=i, total_loss=0.5))
        assert db.is_converged(tol=1e-3)

    def test_loss_decreases(self, tmpdir):
        """Verify loss generally decreases over evolution."""
        from quasim.ciir.simulation.engine import (SimulationConfig,
                                                   run_and_capture_simulation)

        engine = run_and_capture_simulation(
            config=SimulationConfig(
                rep_dim=4,
                batch_size=2,
                n_constraints=2,
                n_observers=1,
                n_steps=50,
                screenshot_interval=0,
                output_dir=tmpdir,
            ),
        )

        losses = engine.dashboard.losses()
        assert len(losses) == 50
        # First loss should be > last loss (evolution reduces loss)
        assert losses[-1] < losses[0]

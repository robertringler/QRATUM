r"""Core simulation engine: orchestrates CIIR → QuASIM → QRATUM pipeline.

Provides:
    CIIRSimulationEngine — Full pipeline with evolution, visualization, and export
    run_and_capture_simulation — Automated batch execution with screenshot capture

Mathematical Pipeline Per Step
------------------------------
1. State initialization:  X ∈ R^{B×R×D} → ρ = X^T X / Tr(X^T X)
2. Loss evaluation:       L(ρ) = Σ λ_i C_i(ρ)² + Σ μ_j |⟨O_j⟩ - y_j|² + γ·Ω(ρ)
3. Gradient step:         ρ_{t+1} = Π_C(ρ_t − η ∇L(ρ_t))
4. Metrics collection:    Loss, ‖∇L‖, C_i(ρ), ⟨O_j⟩, S(ρ), Tr(ρ²)
5. Screenshot capture:    At configured intervals
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from quasim.ciir.config import CIIRConfig
from quasim.ciir.evolution import project_density
from quasim.ciir.loss import CIIRLoss
from quasim.ciir.simulation.ciir_module import (CIIRObserver, CIIRState,
                                                CIIRVisualizer)
from quasim.ciir.simulation.dashboard import (MetricsDashboard,
                                              MetricsSnapshot, print_dashboard)
from quasim.ciir.simulation.export import SimulationExporter
from quasim.ciir.simulation.interactive import InteractiveController
from quasim.ciir.simulation.qratum_module import (PerformanceDashboard,
                                                  QRATUMHardware)
from quasim.ciir.simulation.quasim_module import (TensorRuntime,
                                                  TensorVisualizer)
from quasim.ciir.simulation.renderer import SimulationRenderer
from quasim.ciir.theory import CIIRTheory, build_default_theory


@dataclass
class SimulationConfig:
    """Configuration for the simulation engine."""
    # Manifold dimensions
    rank: int = 4
    rep_dim: int = 8
    batch_size: int = 4
    n_constraints: int = 3
    n_observers: int = 2

    # Evolution
    n_steps: int = 200
    learning_rate: float = 0.01
    entropy_weight: float = 0.01
    convergence_tol: float = 1e-6
    method: str = "projected_gradient"

    # Screenshots
    screenshot_interval: int = 50
    screenshot_dpi: int = 150

    # Export
    output_dir: str = "ciir_output"
    log_tensors: bool = True
    log_metrics: bool = True

    seed: int = 42


class CIIRSimulationEngine:
    r"""Core simulation engine orchestrating the full CIIR → QuASIM → QRATUM pipeline.

    Lifecycle:
        1. init()       — Build CIIR theory, create modules, set up export
        2. run()        — Execute main loop (or step() for single-step control)
        3. shutdown()   — Export final results

    Or use the convenience function:
        run_and_capture_simulation(config)

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    """

    def __init__(self, config: SimulationConfig | None = None):
        if config is None:
            config = SimulationConfig()
        self.config = config
        self.current_step = 0
        self.running = False

        # Modules (initialised in init())
        self.theory: CIIRTheory | None = None
        self.loss_fn: CIIRLoss | None = None
        self.ciir_state: CIIRState | None = None
        self.ciir_observer: CIIRObserver | None = None
        self.ciir_viz: CIIRVisualizer | None = None
        self.tensor_runtime: TensorRuntime | None = None
        self.tensor_viz: TensorVisualizer | None = None
        self.hardware: QRATUMHardware | None = None
        self.dashboard: MetricsDashboard | None = None
        self.controller: InteractiveController | None = None
        self.renderer: SimulationRenderer | None = None
        self.exporter: SimulationExporter | None = None

        # State history for trajectory visualization
        self._state_history: list[CIIRState] = []

    def init(self) -> None:
        """Initialize all subsystems."""
        cfg = self.config

        # 1. Build CIIR theory
        self.theory = build_default_theory(
            rank=cfg.rank,
            rep_dim=cfg.rep_dim,
            n_constraints=cfg.n_constraints,
            n_observers=cfg.n_observers,
            seed=cfg.seed,
        )

        # 2. Create loss function
        self.loss_fn = CIIRLoss(
            theory=self.theory,
            entropy_weight=cfg.entropy_weight,
        )

        # 3. Initialize state
        rng = np.random.default_rng(cfg.seed)
        rho0 = self.theory.manifold.density_matrix(
            self.theory.manifold.random_state(batch=cfg.batch_size, rng=rng)
        )
        self.ciir_state = CIIRState(rho=rho0, theory=self.theory, step=0)

        # 4. Observer
        self.ciir_observer = CIIRObserver(theory=self.theory)

        # 5. Visualizers
        self.ciir_viz = CIIRVisualizer(dpi=cfg.screenshot_dpi)
        self.tensor_viz = TensorVisualizer(dpi=cfg.screenshot_dpi)

        # 6. Tensor runtime
        self.tensor_runtime = TensorRuntime(
            theory=self.theory,
            loss_fn=self.loss_fn,
        )

        # 7. QRATUM hardware
        self.hardware = QRATUMHardware(
            theory=self.theory,
            config=CIIRConfig(
                rank=cfg.rank,
                rep_dim=cfg.rep_dim,
                batch_size=cfg.batch_size,
                n_steps=cfg.n_steps,
                lr=cfg.learning_rate,
                entropy_weight=cfg.entropy_weight,
                seed=cfg.seed,
            ),
        )

        # 8. Dashboard and controller
        self.dashboard = MetricsDashboard()
        self.controller = InteractiveController(
            n_constraints=cfg.n_constraints,
            n_observers=cfg.n_observers,
            learning_rate=cfg.learning_rate,
            entropy_weight=cfg.entropy_weight,
        )

        # 9. Renderer
        self.renderer = SimulationRenderer(dpi=cfg.screenshot_dpi)

        # 10. Exporter
        self.exporter = SimulationExporter(output_dir=cfg.output_dir, dpi=cfg.screenshot_dpi)
        csv_fields = [
            "step", "total_loss", "constraint_loss", "observer_loss",
            "entropy_loss", "gradient_norm", "purity", "entropy", "step_time_ms",
        ]
        self.exporter.init(fieldnames=csv_fields)

        self._state_history = [self.ciir_state]
        self.running = True

    def step(self) -> MetricsSnapshot:
        """Execute a single simulation step.

        Returns
        -------
        snapshot : MetricsSnapshot
            Metrics for this step.
        """
        assert self.ciir_state is not None
        assert self.tensor_runtime is not None
        assert self.hardware is not None

        t0 = time.perf_counter()

        # 1. Compute loss and gradient
        lr = self.controller.learning_rate if self.controller else self.config.learning_rate
        loss, grad, components = self.tensor_runtime.compute_loss_and_gradient(
            self.ciir_state.rho)

        # 2. Gradient step
        rho_new = self.ciir_state.rho - lr * grad

        # 3. Project onto density operator cone
        rho_new = project_density(rho_new)

        # 4. QRATUM hardware constraint injection
        rho_new = self.hardware.inject_constraints(rho_new)

        # 5. Update state
        self.current_step += 1
        self.ciir_state = CIIRState(
            rho=rho_new, theory=self.theory, step=self.current_step)

        # Keep bounded history for trajectory visualization
        self._state_history.append(self.ciir_state)
        if len(self._state_history) > 200:
            self._state_history = self._state_history[-200:]

        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        grad_norm = float(np.linalg.norm(grad))

        # Build metrics snapshot
        snapshot = MetricsSnapshot(
            step=self.current_step,
            total_loss=loss,
            constraint_loss=components.get("constraint", 0.0),
            observer_loss=components.get("observer", 0.0),
            entropy_loss=components.get("entropy", 0.0),
            gradient_norm=grad_norm,
            step_time_ms=step_ms,
            purity=float(self.ciir_state.purity[0]),
            entropy=float(self.ciir_state.entropy[0]),
            constraint_violations=self.ciir_state.constraint_values[0].tolist(),
            observer_expectations=self.ciir_observer.expectation_values(
                self.ciir_state)[0].tolist() if self.ciir_observer else [],
        )

        # Record to dashboard
        if self.dashboard:
            self.dashboard.record(snapshot)

        # Record to QRATUM hardware
        cv = np.array(snapshot.constraint_violations)
        self.hardware.record_step_metrics(
            step=self.current_step,
            loss=loss,
            grad_norm=grad_norm,
            constraint_norms=cv,
            elapsed_s=step_ms / 1000.0,
        )

        return snapshot

    def run(self) -> None:
        """Execute the full simulation loop."""
        cfg = self.config

        for _ in range(cfg.n_steps):
            if not self.running:
                break

            snapshot = self.step()

            # Log metrics to CSV
            if self.exporter and self.exporter.metrics_writer and cfg.log_metrics:
                self.exporter.metrics_writer.write_row({
                    "step": snapshot.step,
                    "total_loss": snapshot.total_loss,
                    "constraint_loss": snapshot.constraint_loss,
                    "observer_loss": snapshot.observer_loss,
                    "entropy_loss": snapshot.entropy_loss,
                    "gradient_norm": snapshot.gradient_norm,
                    "purity": snapshot.purity,
                    "entropy": snapshot.entropy,
                    "step_time_ms": snapshot.step_time_ms,
                })

            # Log tensor state
            if self.exporter and self.exporter.tensor_logger and cfg.log_tensors:
                self.exporter.tensor_logger.log(snapshot.step, {
                    "loss": snapshot.total_loss,
                    "grad_norm": snapshot.gradient_norm,
                    "purity": snapshot.purity,
                    "entropy": snapshot.entropy,
                    "constraint_violations": snapshot.constraint_violations,
                    "observer_expectations": snapshot.observer_expectations,
                })

            # Screenshot at intervals
            if (cfg.screenshot_interval > 0 and
                    self.current_step % cfg.screenshot_interval == 0):
                self._capture_screenshot()

            # Console output
            if self.current_step % 50 == 0 or self.current_step == 1:
                print_dashboard(snapshot)

            # Convergence check
            if self.dashboard and self.dashboard.is_converged(cfg.convergence_tol):
                print(f"\n[Engine] Converged at step {self.current_step}")
                break

    def _capture_screenshot(self) -> None:
        """Render and capture a screenshot of the current simulation state."""
        if not self.renderer or not self.exporter or not self.exporter.screenshot_capture:
            return

        fig = self.renderer.render_frame(
            states=self._state_history,
            runtime=self.tensor_runtime,
            hardware=self.hardware,
        )
        path = self.exporter.screenshot_capture.capture(
            fig, prefix="sim", step=self.current_step)
        print(f"  [Screenshot] {path}")

    def shutdown(self) -> None:
        """Finalize: export all results."""
        self.running = False

        if self.exporter:
            self.exporter.finalize()

        # Generate final dashboard plot
        if self.dashboard and self.config.output_dir:
            dashboard_path = os.path.join(self.config.output_dir, "dashboard.png")
            self.dashboard.plot(save_path=dashboard_path)

        # Generate final performance dashboard
        if self.hardware and self.config.output_dir:
            perf_dashboard = PerformanceDashboard()
            perf_path = os.path.join(
                self.config.output_dir, "performance_dashboard.png")
            perf_dashboard.plot_full_dashboard(self.hardware, save_path=perf_path)

        print(f"\n[Engine] Output: {self.config.output_dir}/")
        print("  screenshots/         — PNG frames")
        print("  metrics.csv          — Per-step metrics")
        print("  tensor_log.json      — Tensor states")
        print("  dashboard.png        — Metrics dashboard")
        print("  performance_dashboard.png — Performance dashboard")


def run_and_capture_simulation(
    config: SimulationConfig | None = None,
    num_steps: int | None = None,
    screenshot_interval: int | None = None,
    output_folder: str | None = None,
) -> CIIRSimulationEngine:
    r"""Run full automated simulation with screenshot capture.

    Primary entry point for batch/automated execution:
      1. Initialize the CIIR → QuASIM → QRATUM pipeline
      2. Step through evolution for num_steps
      3. At each screenshot_interval: render + capture PNG
      4. Log per-step metrics to CSV
      5. Export final tensor log (JSON), metrics (CSV), dashboard (PNG)

    Parameters
    ----------
    config : SimulationConfig, optional
        Full configuration (overrides below are applied on top).
    num_steps : int, optional
        Override number of evolution steps.
    screenshot_interval : int, optional
        Override screenshot interval (0 to disable).
    output_folder : str, optional
        Override output directory.

    Returns
    -------
    engine : CIIRSimulationEngine
        The engine instance (for post-hoc inspection of state/metrics).

    Examples
    --------
    >>> engine = run_and_capture_simulation(
    ...     num_steps=100,
    ...     screenshot_interval=25,
    ...     output_folder="/tmp/ciir_test",
    ... )
    >>> len(engine.dashboard.history)
    100
    """
    if config is None:
        config = SimulationConfig()

    # Apply overrides
    if num_steps is not None:
        config.n_steps = num_steps
    if screenshot_interval is not None:
        config.screenshot_interval = screenshot_interval
    if output_folder is not None:
        config.output_dir = output_folder

    print("╔══════════════════════════════════════════════════╗")
    print("║  CIIR → QuASIM → QRATUM Simulation Engine       ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Manifold: R={config.rank} D={config.rep_dim} B={config.batch_size}")
    print(f"║  Theory:   C={config.n_constraints} O={config.n_observers}"
          f" η={config.learning_rate} γ={config.entropy_weight}")
    print(f"║  Steps:    {config.n_steps}  Screenshots: every {config.screenshot_interval}")
    print(f"║  Output:   {config.output_dir}")
    print("╚══════════════════════════════════════════════════╝\n")

    engine = CIIRSimulationEngine(config)
    engine.init()
    engine.run()
    engine.shutdown()

    return engine

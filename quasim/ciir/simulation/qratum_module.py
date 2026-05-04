r"""QRATUM Hardware Abstraction Module: scheduling, metrics, and dashboards.

Implements the QRATUM layer of the simulation pipeline:

    QRATUMHardware — Hardware control abstraction with virtualized layers
    ExecutionScheduler — Task scheduling and prioritization
    PerformanceDashboard — Metrics visualization and logging

QRATUM Subsystem Mappings
--------------------------
HCAL → hardware constraint injection Π_C
qstack → constraint weighting λ_i
qnx → distributed evolution executor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quasim.ciir.config import CIIRConfig
from quasim.ciir.integration.hcal_bridge import HCALConstraintInjector
from quasim.ciir.integration.qstack_bridge import QStackConstraintWeighter
from quasim.ciir.theory import CIIRTheory, RealTensor

# ================================================================
# QRATUMHardware — Hardware control abstraction
# ================================================================


@dataclass
class QRATUMHardware:
    """Hardware control abstraction for CIIR simulation.

    Virtualizes hardware layers (HCAL, qstack, qnx) and tracks
    device-level metrics for the performance dashboard.

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple.
    config : CIIRConfig
        Simulation configuration.
    """

    theory: CIIRTheory
    config: CIIRConfig
    hcal: HCALConstraintInjector | None = None
    qstack: QStackConstraintWeighter | None = None

    # Device metrics
    _metrics_log: list[dict[str, Any]] = field(default_factory=list)
    _step_times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._metrics_log = []
        self._step_times = []
        self.hcal = HCALConstraintInjector(
            theory=self.theory,
            device_precision=self.config.precision,
            memory_budget_mb=1024.0,
        )
        self.qstack = QStackConstraintWeighter(
            theory=self.theory,
            base_priority=1.0,
        )

    def inject_constraints(self, rho: RealTensor) -> RealTensor:
        """Apply hardware-aware constraint projection via HCAL."""
        assert self.hcal is not None
        return self.hcal.inject_constraints(rho)

    def record_step_metrics(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        constraint_norms: RealTensor,
        elapsed_s: float,
    ) -> None:
        """Record per-step metrics for the dashboard.

        Parameters
        ----------
        step : int
            Evolution step index.
        loss : float
            Total loss value.
        grad_norm : float
            ‖∇L‖.
        constraint_norms : array
            Per-constraint violation norms.
        elapsed_s : float
            Wall-clock time for this step.
        """
        self._step_times.append(elapsed_s)
        self._metrics_log.append(
            {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "constraint_violations": (
                    constraint_norms.tolist()
                    if isinstance(constraint_norms, np.ndarray)
                    else constraint_norms
                ),
                "elapsed_s": elapsed_s,
                "cumulative_time_s": sum(self._step_times),
                "throughput_steps_per_s": (step + 1) / max(sum(self._step_times), 1e-12),
            }
        )

    @property
    def metrics_log(self) -> list[dict[str, Any]]:
        return self._metrics_log

    def device_summary(self) -> dict[str, Any]:
        """Summary of hardware configuration and performance."""
        assert self.hcal is not None
        assert self.qstack is not None
        total_time = sum(self._step_times) if self._step_times else 0.0
        n_steps = len(self._step_times)
        return {
            "hcal": self.hcal.to_hcal_config(),
            "qstack": self.qstack.to_qstack_config(),
            "performance": {
                "total_steps": n_steps,
                "total_time_s": round(total_time, 4),
                "avg_step_time_s": round(total_time / max(n_steps, 1), 6),
                "throughput_steps_per_s": round(n_steps / max(total_time, 1e-12), 2),
            },
        }


# ================================================================
# ExecutionScheduler — Task scheduling and prioritization
# ================================================================


@dataclass
class ExecutionScheduler:
    """Task scheduling and prioritization for CIIR evolution.

    Manages the execution order and priority of constraint/observer
    evaluations based on qstack weights and convergence history.

    Parameters
    ----------
    hardware : QRATUMHardware
        The hardware abstraction.
    """

    hardware: QRATUMHardware
    _task_queue: list[dict[str, Any]] = field(default_factory=list)
    _completed: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._task_queue = []
        self._completed = []

    def schedule_evolution(
        self, n_steps: int, lr: float, method: str = "projected_gradient"
    ) -> str:
        """Schedule an evolution run.

        Returns
        -------
        task_id : str
        """
        task_id = f"ciir-evo-{len(self._task_queue)}"
        assert self.hardware.qstack is not None
        self._task_queue.append(
            {
                "task_id": task_id,
                "type": "evolution",
                "n_steps": n_steps,
                "lr": lr,
                "method": method,
                "status": "queued",
                "priority": self.hardware.qstack.base_priority,
                "constraint_priorities": self.hardware.qstack.constraint_priorities(),
            }
        )
        return task_id

    def schedule_measurement(self, observer_index: int) -> str:
        """Schedule a measurement task.

        Returns
        -------
        task_id : str
        """
        task_id = f"ciir-meas-{len(self._task_queue)}"
        self._task_queue.append(
            {
                "task_id": task_id,
                "type": "measurement",
                "observer_index": observer_index,
                "status": "queued",
                "priority": 2.0,  # Measurements get high priority
            }
        )
        return task_id

    def next_task(self) -> dict[str, Any] | None:
        """Get next task from queue (highest priority first)."""
        if not self._task_queue:
            return None
        # Sort by priority descending
        self._task_queue.sort(key=lambda t: t.get("priority", 0), reverse=True)
        task = self._task_queue.pop(0)
        task["status"] = "running"
        return task

    def complete_task(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        """Mark a task as completed."""
        task["status"] = "completed"
        task["result"] = result
        self._completed.append(task)

    @property
    def pending_count(self) -> int:
        return len(self._task_queue)

    @property
    def completed_tasks(self) -> list[dict[str, Any]]:
        return self._completed


# ================================================================
# PerformanceDashboard — Metrics visualization and logging
# ================================================================


class PerformanceDashboard:
    """Real-time performance metrics dashboard.

    Generates visualizations of:
    - Loss convergence with component breakdown
    - Constraint violation norms over time
    - Gradient magnitude evolution
    - Step timing and throughput
    - Hardware utilization
    """

    def __init__(self, figsize: tuple[int, int] = (16, 10), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def plot_full_dashboard(
        self,
        hardware: QRATUMHardware,
        save_path: str | None = None,
    ) -> Any:
        """Generate comprehensive performance dashboard.

        Parameters
        ----------
        hardware : QRATUMHardware
            Hardware with recorded metrics.
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        metrics = hardware.metrics_log
        if not metrics:
            return None

        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)

        steps = [m["step"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        grad_norms = [m["grad_norm"] for m in metrics]
        step_times = [m["elapsed_s"] for m in metrics]
        throughputs = [m["throughput_steps_per_s"] for m in metrics]

        # 1. Loss convergence
        ax = axes[0, 0]
        ax.plot(steps, losses, "b-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Convergence")
        ax.grid(True, alpha=0.3)

        # 2. Gradient norms
        ax = axes[0, 1]
        ax.semilogy(steps, grad_norms, "r-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("‖∇L‖")
        ax.set_title("Gradient Magnitude")
        ax.grid(True, alpha=0.3)

        # 3. Constraint violations
        ax = axes[0, 2]
        cv_data = [m["constraint_violations"] for m in metrics]
        if cv_data and cv_data[0]:
            n_c = len(cv_data[0])
            for c_idx in range(n_c):
                values = [cv[c_idx] if c_idx < len(cv) else 0 for cv in cv_data]
                ax.plot(steps, values, linewidth=1, label=f"C{c_idx}")
            ax.legend(fontsize=7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Violation Norm")
        ax.set_title("Constraint Violations")
        ax.grid(True, alpha=0.3)

        # 4. Step timing
        ax = axes[1, 0]
        ax.plot(steps, [t * 1000 for t in step_times], "g-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Step Latency")
        ax.grid(True, alpha=0.3)

        # 5. Throughput
        ax = axes[1, 1]
        ax.plot(steps, throughputs, "m-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Steps/s")
        ax.set_title("Throughput")
        ax.grid(True, alpha=0.3)

        # 6. Hardware summary text
        ax = axes[1, 2]
        ax.axis("off")
        summary = hardware.device_summary()
        text_lines = [
            "QRATUM Hardware Summary",
            "=" * 30,
            f"Precision: {summary['hcal']['precision']}",
            f"Max batch: {summary['hcal']['max_batch']}",
            f"Constraints: {summary['hcal']['n_constraints']}",
            f"Observers: {summary['hcal']['n_observers']}",
            "",
            f"Total steps: {summary['performance']['total_steps']}",
            f"Total time: {summary['performance']['total_time_s']:.3f}s",
            f"Avg step: {summary['performance']['avg_step_time_s'] * 1000:.2f}ms",
            f"Throughput: {summary['performance']['throughput_steps_per_s']:.1f} steps/s",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            verticalalignment="top",
        )

        plt.suptitle("CIIR → QuASIM → QRATUM Performance Dashboard", fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_convergence_rate(
        self,
        hardware: QRATUMHardware,
        save_path: str | None = None,
    ) -> Any:
        """Plot convergence rate: |ΔL|/L vs step.

        Parameters
        ----------
        hardware : QRATUMHardware
        save_path : str, optional
        """
        import matplotlib.pyplot as plt

        metrics = hardware.metrics_log
        if len(metrics) < 2:
            return None

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        losses = [m["loss"] for m in metrics]
        rates = []
        for i in range(1, len(losses)):
            delta = abs(losses[i] - losses[i - 1])
            rate = delta / max(abs(losses[i - 1]), 1e-12)
            rates.append(rate)

        ax.semilogy(range(1, len(losses)), rates, "b-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("|ΔL| / |L|")
        ax.set_title("Convergence Rate")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

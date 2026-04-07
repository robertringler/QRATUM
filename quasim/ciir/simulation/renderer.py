"""3D renderer using matplotlib for CIIR simulation visualization.

Provides:
    SimulationRenderer — Orchestrates all 3D scene rendering
    render_simulation_frame — Render a complete frame with all panels
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quasim.ciir.simulation.ciir_module import CIIRState, CIIRObserver, CIIRVisualizer
from quasim.ciir.simulation.quasim_module import TensorRuntime, TensorVisualizer
from quasim.ciir.simulation.qratum_module import PerformanceDashboard, QRATUMHardware


class SimulationRenderer:
    """Orchestrates all 3D scene rendering for the simulation.

    Generates composite frames with:
    - CIIR manifold trajectory (top-left)
    - Constraint surface heatmap (top-right)
    - Loss landscape / gradient norms (bottom-left)
    - Performance dashboard (bottom-right)

    Parameters
    ----------
    figsize : tuple
        Overall figure size in inches.
    dpi : int
        Resolution.
    """

    def __init__(self, figsize: tuple[int, int] = (20, 14), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.ciir_viz = CIIRVisualizer(figsize=(10, 7), dpi=dpi)
        self.tensor_viz = TensorVisualizer(figsize=(10, 7), dpi=dpi)
        self.dashboard = PerformanceDashboard(figsize=(16, 10), dpi=dpi)

    def render_frame(
        self,
        states: list[CIIRState],
        runtime: TensorRuntime,
        hardware: QRATUMHardware,
        save_path: str | None = None,
    ) -> Any:
        """Render a complete simulation frame with 4 panels.

        Parameters
        ----------
        states : list of CIIRState
            Evolution trajectory (at least 1 state).
        runtime : TensorRuntime
            Tensor runtime with contraction log.
        hardware : QRATUMHardware
            Hardware with metrics log.
        save_path : str, optional
            Path to save the composite figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        # ---- Panel 1: CIIR manifold trajectory (top-left) ----
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        _plot_manifold_on_ax(ax1, states)

        # ---- Panel 2: Constraint surface (top-right) ----
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        _plot_constraints_on_ax(ax2, states[-1])

        # ---- Panel 3: Loss convergence + gradient (bottom-left) ----
        ax3 = fig.add_subplot(2, 2, 3)
        _plot_loss_on_ax(ax3, runtime.contraction_log)

        # ---- Panel 4: Metrics summary (bottom-right) ----
        ax4 = fig.add_subplot(2, 2, 4)
        _plot_metrics_on_ax(ax4, hardware, states[-1])

        step = states[-1].step if states else 0
        fig.suptitle(
            f"CIIR → QuASIM → QRATUM Simulation  |  Step {step}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return fig


def render_simulation_frame(
    states: list[CIIRState],
    runtime: TensorRuntime,
    hardware: QRATUMHardware,
    save_path: str | None = None,
    dpi: int = 150,
) -> Any:
    """Convenience function to render a single composite frame."""
    renderer = SimulationRenderer(dpi=dpi)
    return renderer.render_frame(states, runtime, hardware, save_path)


# ================================================================
# Internal panel renderers
# ================================================================


def _plot_manifold_on_ax(ax: Any, states: list[CIIRState]) -> None:
    """Plot 3D manifold trajectory on a given axes."""
    for b in range(states[0].batch_size):
        coords = np.array([s.manifold_embedding_3d()[b] for s in states])
        colors = np.linspace(0, 1, len(coords))
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=colors, cmap="viridis", s=8, alpha=0.7,
        )
        ax.plot(
            coords[:, 0], coords[:, 1], coords[:, 2],
            alpha=0.3, linewidth=0.5,
        )
    ax.set_xlabel("λ₁")
    ax.set_ylabel("λ₂")
    ax.set_zlabel("λ₃")
    ax.set_title("CIIR Manifold Trajectory")


def _plot_constraints_on_ax(ax: Any, state: CIIRState) -> None:
    """Plot constraint surface heatmap on a given axes."""
    points, violations = state.constraint_surface_samples(100)
    v_norm = violations / max(violations.max(), 1e-12)
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=v_norm, cmap="hot_r", s=15, alpha=0.6,
    )
    current = state.manifold_embedding_3d()
    ax.scatter(
        current[:, 0], current[:, 1], current[:, 2],
        c="blue", s=80, marker="*",
    )
    ax.set_xlabel("λ₁")
    ax.set_ylabel("λ₂")
    ax.set_zlabel("λ₃")
    ax.set_title("Constraint Surface")


def _plot_loss_on_ax(ax: Any, contraction_log: list[dict]) -> None:
    """Plot loss convergence on a given axes."""
    if not contraction_log:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return
    steps = range(len(contraction_log))
    losses = [e["loss"] for e in contraction_log]
    ax.plot(steps, losses, "b-", linewidth=1, label="Total Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.grid(True, alpha=0.3)

    # Secondary y-axis for gradient norm
    ax2 = ax.twinx()
    grad_norms = [e["grad_norm"] for e in contraction_log]
    ax2.semilogy(steps, grad_norms, "r-", linewidth=0.5, alpha=0.5, label="‖∇L‖")
    ax2.set_ylabel("‖∇L‖", color="red")


def _plot_metrics_on_ax(ax: Any, hardware: QRATUMHardware, state: CIIRState) -> None:
    """Plot metrics summary as text on a given axes."""
    ax.axis("off")
    summary = hardware.device_summary()
    perf = summary.get("performance", {})

    lines = [
        "QRATUM Hardware Summary",
        "=" * 35,
        f"Precision:   {summary.get('hcal', {}).get('precision', 'N/A')}",
        f"Constraints: {summary.get('hcal', {}).get('n_constraints', 0)}",
        f"Observers:   {summary.get('hcal', {}).get('n_observers', 0)}",
        "",
        f"Total steps:  {perf.get('total_steps', 0)}",
        f"Total time:   {perf.get('total_time_s', 0):.3f}s",
        f"Avg step:     {perf.get('avg_step_time_s', 0) * 1000:.2f}ms",
        f"Throughput:   {perf.get('throughput_steps_per_s', 0):.1f} steps/s",
        "",
        f"Current state (batch 0):",
        f"  Purity:   {float(state.purity[0]):.4f}",
        f"  Entropy:  {float(state.entropy[0]):.4f}",
    ]
    # Add constraint violations
    cv = state.constraint_values
    if cv.shape[1] > 0:
        for i in range(cv.shape[1]):
            lines.append(f"  C{i} value: {float(cv[0, i]):.6f}")

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=8, family="monospace",
        verticalalignment="top",
    )
    ax.set_title("Metrics Dashboard")

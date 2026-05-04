"""Real-time metrics dashboard for CIIR simulation.

Provides:
    MetricsDashboard — Track and visualize simulation metrics
    print_dashboard — Console-based dashboard for headless mode
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsSnapshot:
    """Single-step metrics snapshot."""

    step: int = 0
    total_loss: float = 0.0
    constraint_loss: float = 0.0
    observer_loss: float = 0.0
    entropy_loss: float = 0.0
    gradient_norm: float = 0.0
    step_time_ms: float = 0.0
    purity: float = 0.0
    entropy: float = 0.0
    constraint_violations: list[float] = field(default_factory=list)
    observer_expectations: list[float] = field(default_factory=list)


class MetricsDashboard:
    """Track and visualize simulation metrics over time.

    Stores a rolling history of metrics for plotting and
    console display.

    Parameters
    ----------
    history_size : int
        Maximum number of snapshots to retain.
    """

    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self._history: list[MetricsSnapshot] = []

    def record(self, snapshot: MetricsSnapshot) -> None:
        """Record a new metrics snapshot."""
        self._history.append(snapshot)
        if len(self._history) > self.history_size:
            self._history.pop(0)

    @property
    def history(self) -> list[MetricsSnapshot]:
        return self._history

    @property
    def latest(self) -> MetricsSnapshot | None:
        return self._history[-1] if self._history else None

    def losses(self) -> list[float]:
        return [s.total_loss for s in self._history]

    def gradient_norms(self) -> list[float]:
        return [s.gradient_norm for s in self._history]

    def is_converged(self, tol: float = 1e-6, window: int = 10) -> bool:
        """Check if loss has converged over last `window` steps."""
        if len(self._history) < window:
            return False
        recent = [s.total_loss for s in self._history[-window:]]
        return max(recent) - min(recent) < tol

    def plot(self, save_path: str | None = None) -> Any:
        """Plot metrics dashboard.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)

        steps = [s.step for s in self._history]
        losses = [s.total_loss for s in self._history]
        grad_norms = [s.gradient_norm for s in self._history]
        purities = [s.purity for s in self._history]
        entropies = [s.entropy for s in self._history]

        # Loss
        ax = axes[0, 0]
        ax.plot(steps, losses, "b-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Convergence")
        ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = axes[0, 1]
        ax.semilogy(steps, grad_norms, "r-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("‖∇L‖")
        ax.set_title("Gradient Magnitude")
        ax.grid(True, alpha=0.3)

        # Purity and entropy
        ax = axes[1, 0]
        ax.plot(steps, purities, "g-", linewidth=1, label="Purity Tr(ρ²)")
        ax.plot(steps, entropies, "m-", linewidth=1, label="Entropy S(ρ)")
        ax.set_xlabel("Step")
        ax.legend()
        ax.set_title("State Properties")
        ax.grid(True, alpha=0.3)

        # Constraint violations
        ax = axes[1, 1]
        if self._history and self._history[0].constraint_violations:
            n_c = len(self._history[0].constraint_violations)
            for c_idx in range(n_c):
                vals = [
                    s.constraint_violations[c_idx] if c_idx < len(s.constraint_violations) else 0
                    for s in self._history
                ]
                ax.plot(steps, vals, linewidth=1, label=f"C{c_idx}")
            ax.legend(fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("C_i(ρ)")
        ax.set_title("Constraint Values")
        ax.grid(True, alpha=0.3)

        plt.suptitle("CIIR Simulation Dashboard", fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return fig


def print_dashboard(snapshot: MetricsSnapshot, file: Any = None) -> None:
    """Print compact dashboard to console.

    Parameters
    ----------
    snapshot : MetricsSnapshot
    file : file-like, optional
        Output stream (default: sys.stdout).
    """
    if file is None:
        file = sys.stdout

    cv_str = " ".join(f"{v:.4f}" for v in snapshot.constraint_violations[:5])
    obs_str = " ".join(f"{v:.4f}" for v in snapshot.observer_expectations[:5])

    print(
        f"Step {snapshot.step:>5d} | "
        f"Loss {snapshot.total_loss:>9.4f} | "
        f"‖∇L‖ {snapshot.gradient_norm:>9.4e} | "
        f"Purity {snapshot.purity:.4f} | "
        f"S {snapshot.entropy:.4f} | "
        f"C: [{cv_str}] | "
        f"O: [{obs_str}] | "
        f"{snapshot.step_time_ms:.1f}ms",
        file=file,
    )

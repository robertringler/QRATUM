"""Visualization Agent — plots for the CIIR multi-qubit controller."""

from .plots import (plot_all, plot_bloch_trajectory, plot_entanglement_entropy,
                    plot_error_rate_recursion, plot_lyapunov,
                    plot_phase_diagram)

__all__ = [
    "plot_bloch_trajectory",
    "plot_entanglement_entropy",
    "plot_error_rate_recursion",
    "plot_phase_diagram",
    "plot_lyapunov",
    "plot_all",
]

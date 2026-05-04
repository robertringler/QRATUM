"""Visualization Agent — generates all required plots.

Plots produced
--------------
1. Bloch sphere trajectory (single-qubit reduced state)
2. Entanglement entropy vs time
3. Logical error rate vs recursion depth
4. Phase diagram: p_phys vs p_L^final (threshold visible)
5. Lyapunov V(t) and dV/dt vs time

All plots are saved to ``results/`` directory (created if needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

# matplotlib backend must be set before pyplot import
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CMatrix = NDArray[np.complex128]
RESULTS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "results" / "multi_qubit"


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _bloch_coords(rho: CMatrix) -> Tuple[float, float, float]:
    """Extract Bloch vector (x, y, z) from a 2×2 density matrix."""
    sx = 2.0 * float(np.real(rho[0, 1]))
    sy = 2.0 * float(np.imag(rho[1, 0]))
    sz = float(np.real(rho[0, 0] - rho[1, 1]))
    return sx, sy, sz


def _reduced_qubit(rho: CMatrix, n_qubits: int, qubit: int = 0) -> CMatrix:
    """Partial trace to get single-qubit reduced state."""
    from quasim.ciir.multi_qubit.quantum.density_matrix import _partial_trace

    return _partial_trace(rho, n_qubits, [qubit])


def plot_bloch_trajectory(
    rho_traj: List[CMatrix],
    n_qubits: int,
    qubit: int = 0,
    title: str = "Bloch Sphere Trajectory",
    save: bool = True,
) -> Path:
    """Plot the Bloch sphere trajectory of one qubit."""
    outdir = _ensure_results_dir()
    xs, ys, zs = [], [], []
    for rho in rho_traj:
        rho1 = _reduced_qubit(rho, n_qubits, qubit)
        bx, by, bz = _bloch_coords(rho1)
        xs.append(bx)
        ys.append(by)
        zs.append(bz)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xsph = np.outer(np.cos(u), np.sin(v))
    ysph = np.outer(np.sin(u), np.sin(v))
    zsph = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xsph, ysph, zsph, color="lightgrey", alpha=0.3, linewidth=0.5)

    # Axes
    for vec, label in [([1.1, 0, 0], "x"), ([0, 1.1, 0], "y"), ([0, 0, 1.1], "z")]:
        ax.quiver(0, 0, 0, *vec, color="black", linewidth=1, arrow_length_ratio=0.1)
        ax.text(*[1.15 * v for v in vec], label, fontsize=10)

    n = len(xs)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(xs[i : i + 2], ys[i : i + 2], zs[i : i + 2], color=colors[i], linewidth=1.5)

    ax.scatter(*[xs[-1]], *[ys[-1]], *[zs[-1]], color="red", s=50, zorder=5, label="final")
    ax.scatter(*[xs[0]], *[ys[0]], *[zs[0]], color="blue", s=50, zorder=5, label="initial")
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    out = outdir / "bloch_trajectory.png"
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_entanglement_entropy(
    times: NDArray,
    entropy_traj: NDArray,
    title: str = "Entanglement Entropy vs Time",
    save: bool = True,
) -> Path:
    """Plot S(t) = Von Neumann entropy of reduced qubit state."""
    outdir = _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, entropy_traj, color="steelblue", linewidth=2)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Entanglement Entropy S(ρ_A)")
    ax.set_title(title, fontsize=13)
    ax.axhline(np.log(2), color="red", linestyle="--", alpha=0.6, label="log 2 (max for qubit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = outdir / "entanglement_entropy.png"
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_error_rate_recursion(
    rates_dict: dict,
    p_star: float,
    title: str = "Logical Error Rate vs Recursion Depth",
    save: bool = True,
) -> Path:
    """Plot p_L(n) for several values of p_phys.

    Parameters
    ----------
    rates_dict : {p_phys_label: list_of_rates}
    p_star     : threshold value (vertical reference)
    """
    outdir = _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(rates_dict)))
    for (label, rates), color in zip(rates_dict.items(), colors):
        levels = list(range(len(rates)))
        valid = [(n, r) for n, r in zip(levels, rates) if r > 0]
        if valid:
            ns, rs = zip(*valid)
            ax.semilogy(ns, rs, marker="o", label=f"p={label}", color=color, linewidth=2)
    ax.axhline(p_star, color="black", linestyle="--", linewidth=1.5, label=f"p* = {p_star:.4f}")
    ax.set_xlabel("Concatenation level n")
    ax.set_ylabel("Logical error rate p_L(n)")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    out = outdir / "error_rate_recursion.png"
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_phase_diagram(
    p_phys_arr: NDArray,
    p_L_final_arr: NDArray,
    p_star: float,
    title: str = "Phase Diagram: Threshold Behaviour",
    save: bool = True,
) -> Path:
    """Plot p_L^final vs p_phys, revealing the threshold transition."""
    outdir = _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(p_phys_arr, p_L_final_arr, "b-o", markersize=4, linewidth=2, label="p_L (8 levels)")
    ax.loglog(
        p_phys_arr, p_phys_arr, "k--", linewidth=1.5, alpha=0.5, label="p_L = p_phys (no code)"
    )
    ax.axvline(p_star, color="red", linestyle="--", linewidth=1.5, label=f"p* = {p_star:.4f}")
    ax.fill_betweenx([1e-30, 1], 0, p_star, alpha=0.1, color="green", label="below threshold")
    ax.fill_betweenx([1e-30, 1], p_star, 1, alpha=0.1, color="red", label="above threshold")
    ax.set_xlabel("Physical error rate p_phys")
    ax.set_ylabel("Final logical error rate p_L (level 8)")
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(p_phys_arr.min(), p_phys_arr.max())
    ax.set_ylim(1e-30, 1)
    out = outdir / "phase_diagram.png"
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lyapunov(
    times: NDArray,
    V_traj: NDArray,
    dV_traj: NDArray,
    title: str = "Lyapunov Function V(t)",
    save: bool = True,
) -> Path:
    """Plot V(t) and dV/dt vs time."""
    outdir = _ensure_results_dir()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(times, V_traj, color="darkorange", linewidth=2)
    ax1.set_ylabel("V(x, ρ)")
    ax1.set_title(title, fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="k", linewidth=0.8)

    ax2.plot(times, dV_traj, color="crimson", linewidth=1.5, alpha=0.8)
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("dV/dt")
    ax2.set_xlabel("Time t")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Time derivative dV/dt (should be ≤ 0 for stability)")

    fig.tight_layout()
    out = outdir / "lyapunov.png"
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all(
    sim_result, lyap_result, qec_results: dict, p_phys_arr, p_L_arr, p_star: float
) -> List[Path]:
    """Generate all required plots and return list of output paths."""

    n_q = sim_result.params["n_qubits"]
    paths = []

    paths.append(plot_bloch_trajectory(sim_result.rho_traj, n_q, qubit=0))
    paths.append(plot_entanglement_entropy(sim_result.times, sim_result.entropy_traj))
    paths.append(plot_error_rate_recursion(qec_results, p_star))
    paths.append(plot_phase_diagram(p_phys_arr, p_L_arr, p_star))
    paths.append(
        plot_lyapunov(
            sim_result.times,
            lyap_result.V_traj,
            lyap_result.dV_traj,
        )
    )
    return paths

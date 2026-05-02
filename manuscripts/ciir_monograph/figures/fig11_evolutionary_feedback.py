#!/usr/bin/env python3
"""
Figure 11: Neuro-Evolutionary Feedback Loop of Constraint-Induced Perception
==============================================================================
GPAI STEM VISUALIZER — Prompt:
  "CIIR neuro-evolutionary realism feedback loop: spiral convergence
   diagram showing how constraint-induced interfaces self-reinforce.
   Observers mistake interface utility for ontological truth. Show
   the CRS→CIIR→Observer loop embedding in evolutionary dynamics
   with fitness landscape, selection pressure, and interface lock-in.
   Scientific Diagram Style."

Design choices:
  - Central spiral diagram showing convergence loop
  - Fitness landscape with interface-optimal vs truth-optimal paths
  - Phase portrait of realism-utility coupling dynamics
  - Evolutionary trajectory toward interface lock-in
  - Nature/Science journal colour palette, 300 DPI

Publication-quality, 4K-equivalent at 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ── Colour palette ──
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
C_EVOLVE   = "#27AE60"   # evolution / fitness
C_REALISM  = "#E74C3C"   # realism error
C_UTILITY  = "#2980B9"   # utility / interface
C_LOCK     = "#8E44AD"   # lock-in / fixed point
C_CRS      = "#E67E22"   # CRS layer
C_ARROW    = "#7F8C8D"
C_SPIRAL   = "#1ABC9C"


def main():
    fig = plt.figure(figsize=(20, 14), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════════════════════════
    # Panel A: Spiral convergence diagram
    # ═══════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(221)
    ax1.set_facecolor(C_BG)

    # Draw converging spiral representing iterative loop
    t = np.linspace(0, 6 * np.pi, 1000)
    decay = np.exp(-t / (4 * np.pi))
    r = 1.8 * decay + 0.2

    x_spiral = r * np.cos(t)
    y_spiral = r * np.sin(t)

    # Colour by iteration (early=red, late=green)
    for i in range(len(t) - 1):
        progress = t[i] / t[-1]
        colour = plt.cm.RdYlGn(progress)
        ax1.plot(x_spiral[i:i+2], y_spiral[i:i+2],
                 color=colour, lw=2.5, alpha=0.8)

    # Arrow heads at key points
    for k in [100, 300, 500, 700, 900]:
        dx = x_spiral[k] - x_spiral[k-5]
        dy = y_spiral[k] - y_spiral[k-5]
        ax1.annotate(
            "", xy=(x_spiral[k], y_spiral[k]),
            xytext=(x_spiral[k] - dx*3, y_spiral[k] - dy*3),
            arrowprops=dict(arrowstyle="-|>",
                            color=plt.cm.RdYlGn(t[k] / t[-1]),
                            lw=2.0, mutation_scale=12)
        )

    # Fixed point at center
    fp = Circle((0, 0), 0.25, facecolor=C_LOCK, alpha=0.15,
                edgecolor=C_LOCK, linewidth=3.0, zorder=5)
    ax1.add_patch(fp)
    ax1.text(0, 0.03, "Interface\nLock-in",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=C_LOCK, family="serif", zorder=6)
    ax1.text(0, -0.05, r"$\rho^*$",
             ha="center", va="top", fontsize=14, fontweight="bold",
             color=C_LOCK, family="serif", zorder=6)

    # Loop stage labels around the spiral
    stage_labels = [
        (0.25, "CRS generates\nstructure", C_CRS),
        (1.1, "CIIR constrains\nperception", C_LOCK),
        (2.0, "Observer acts on\ninterface icons", C_UTILITY),
        (2.9, "Selection favours\nutility over truth", C_EVOLVE),
        (3.8, "Constraints\ntighten", C_REALISM),
        (4.7, "Interface\nstabilises", C_SPIRAL),
    ]

    for t_val, label, col in stage_labels:
        # Position label outside spiral
        idx = int(t_val / t[-1] * (len(t) - 1))
        idx = min(idx, len(t) - 1)
        r_label = r[idx] + 0.5
        ax1.text(r_label * np.cos(t[idx]),
                 r_label * np.sin(t[idx]),
                 label, ha="center", va="center",
                 fontsize=7, color=col, family="serif",
                 fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor=col,
                           boxstyle="round,pad=0.15", alpha=0.85,
                           linewidth=0.8))

    ax1.set_xlim(-2.8, 2.8)
    ax1.set_ylim(-2.8, 2.8)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(
        "(a) Convergence Spiral: Evolutionary Interface Lock-in",
        fontsize=12, fontweight="bold", family="serif", pad=10)

    # ═══════════════════════════════════════════════════════════════
    # Panel B: Fitness landscape — veridical vs interface
    # ═══════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.set_facecolor(C_BG)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Fitness landscape: interface strategies have higher fitness
    # than veridical ones
    # Two peaks: one for "truth" (lower), one for "interface" (higher)
    truth_peak = 1.5 * np.exp(-((X + 1.2)**2 + (Y + 1.0)**2) / 1.2)
    interface_peak = 3.0 * np.exp(-((X - 0.8)**2 + (Y - 0.5)**2) / 1.5)
    noise = 0.15 * np.sin(3*X) * np.cos(2*Y)
    Z = truth_peak + interface_peak + noise

    ax2.plot_surface(X, Y, Z, cmap="YlGn", alpha=0.5,
                     edgecolor="none", antialiased=True)

    # Mark the two peaks
    ax2.scatter([-1.2], [-1.0], [1.55], color=C_REALISM, s=100,
                zorder=5, edgecolor="white", linewidth=1.5)
    ax2.text(-1.2, -1.0, 1.8, "Veridical\n(low fitness)",
             fontsize=8, color=C_REALISM, family="serif",
             ha="center", fontweight="bold")

    ax2.scatter([0.8], [0.5], [3.05], color=C_EVOLVE, s=100,
                zorder=5, edgecolor="white", linewidth=1.5)
    ax2.text(0.8, 0.5, 3.3, "Interface\n(high fitness)",
             fontsize=8, color=C_EVOLVE, family="serif",
             ha="center", fontweight="bold")

    # Evolutionary trajectory
    t_traj = np.linspace(0, 1, 50)
    traj_x = -1.2 + 2.0 * t_traj
    traj_y = -1.0 + 1.5 * t_traj
    r_traj = np.sqrt((traj_x + 1.2)**2 + (traj_y + 1.0)**2)
    traj_z = (1.5 * np.exp(-((traj_x + 1.2)**2 + (traj_y + 1.0)**2) / 1.2)
              + 3.0 * np.exp(-((traj_x - 0.8)**2 + (traj_y - 0.5)**2) / 1.5)
              + 0.05)
    ax2.plot(traj_x, traj_y, traj_z, color=C_CRS, lw=3.0, zorder=4)
    # Arrow at end
    ax2.quiver(traj_x[-5], traj_y[-5], traj_z[-5],
               traj_x[-1]-traj_x[-5], traj_y[-1]-traj_y[-5],
               traj_z[-1]-traj_z[-5],
               color=C_CRS, arrow_length_ratio=0.4, linewidth=2)

    ax2.set_xlabel("Veridical fidelity", fontsize=9, family="serif",
                   labelpad=3)
    ax2.set_ylabel("Interface simplicity", fontsize=9, family="serif",
                   labelpad=3)
    ax2.set_zlabel("Fitness $W$", fontsize=9, family="serif", labelpad=3)
    ax2.set_title(
        "(b) Fitness Landscape: Interface Beats Truth",
        fontsize=12, fontweight="bold", family="serif", pad=8)
    ax2.view_init(elev=25, azim=-50)

    # ═══════════════════════════════════════════════════════════════
    # Panel C: Phase portrait — realism vs utility dynamics
    # ═══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(C_BG)

    # Dynamical system:
    # ṗ = p(1-p)(α·u - β·v)   (realism fraction p)
    # u̇ = γ·(1-p) - δ·u       (utility growth)
    # Simplified: phase portrait in (realism_error, utility) space

    re = np.linspace(0.01, 0.99, 20)
    ut = np.linspace(0.01, 2.0, 20)
    RE, UT = np.meshgrid(re, ut)

    # Flow: realism error increases when utility is high
    # Utility increases when interface is adopted
    alpha, beta, gamma = 1.5, 0.5, 1.0
    dRE = alpha * RE * (1 - RE) * UT - beta * RE  # realism error grows with utility
    dUT = gamma * (1 - RE) - 0.3 * UT   # utility grows when veridical is abandoned

    # Normalize for visualization
    speed = np.sqrt(dRE**2 + dUT**2)
    dRE_n = dRE / (speed + 0.01)
    dUT_n = dUT / (speed + 0.01)

    ax3.streamplot(RE, UT, dRE_n, dUT_n, color=speed, cmap="viridis",
                   density=1.5, linewidth=1.0, arrowsize=1.2)

    # Fixed points
    # Find where dRE ≈ 0 and dUT ≈ 0
    fp_re = 0.7  # approximate fixed point
    fp_ut = gamma * (1 - fp_re) / 0.3
    ax3.plot(fp_re, fp_ut, "o", color=C_LOCK, markersize=12,
             zorder=5, markeredgecolor="white", markeredgewidth=2)
    ax3.text(fp_re + 0.05, fp_ut + 0.1, "Interface\nFixed Point",
             fontsize=9, color=C_LOCK, family="serif", fontweight="bold")

    # Unstable fixed point at origin region
    ax3.plot(0.05, 0.1, "x", color=C_REALISM, markersize=10,
             markeredgewidth=2, zorder=5)
    ax3.text(0.10, 0.15, "Veridical\n(unstable)",
             fontsize=8, color=C_REALISM, family="serif")

    ax3.set_xlabel("Realism error $\\epsilon_R$", fontsize=11,
                   family="serif")
    ax3.set_ylabel("Interface utility $U$", fontsize=11, family="serif")
    ax3.set_title(
        r"(c) Phase Portrait: $(\epsilon_R, U)$ Dynamics",
        fontsize=12, fontweight="bold", family="serif", pad=10)
    ax3.grid(True, alpha=0.15)

    # Key equation
    ax3.text(0.55, 1.8,
             r"$\dot{\epsilon}_R = \alpha\,\epsilon_R(1-\epsilon_R)\,U "
             r"- \beta\,\epsilon_R$"
             "\n"
             r"$\dot{U} = \gamma\,(1-\epsilon_R) - \delta\,U$",
             fontsize=8, color=C_TEXT, family="serif",
             bbox=dict(facecolor="white", edgecolor=C_LOCK,
                       boxstyle="round,pad=0.3", linewidth=1.5,
                       alpha=0.95))

    # ═══════════════════════════════════════════════════════════════
    # Panel D: Evolutionary timeline
    # ═══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(C_BG)

    # Simulate realism error and utility over evolutionary time
    n_gen = 200
    t_gen = np.arange(n_gen)

    # Realism error grows sigmoidally
    eps_R = 0.9 / (1 + np.exp(-0.05 * (t_gen - 80)))
    # Interface utility grows, then saturates
    U_val = 2.0 * (1 - np.exp(-0.02 * t_gen))
    # Fitness grows with utility
    fitness = 1.0 + 3.0 * U_val / U_val.max()
    # Veridical fidelity = 1 - eps_R
    veridical = 1.0 - eps_R

    ax4.plot(t_gen, eps_R, color=C_REALISM, lw=2.5,
             label=r"Realism error $\epsilon_R$")
    ax4.plot(t_gen, veridical, color=C_UTILITY, lw=2.5, ls="--",
             label=r"Veridical fidelity $1 - \epsilon_R$")
    ax4.plot(t_gen, fitness / fitness.max(), color=C_EVOLVE, lw=2.5,
             label=r"Normalised fitness $W/W_{\max}$")

    # Phase regions
    ax4.axvspan(0, 50, facecolor=C_EVOLVE, alpha=0.04)
    ax4.axvspan(50, 120, facecolor=C_CRS, alpha=0.04)
    ax4.axvspan(120, 200, facecolor=C_LOCK, alpha=0.04)

    ax4.text(25, 1.05, "Phase I:\nWeak constraints",
             ha="center", va="bottom", fontsize=7, color=C_EVOLVE,
             family="serif", fontweight="bold")
    ax4.text(85, 1.05, "Phase II:\nInterface adoption",
             ha="center", va="bottom", fontsize=7, color=C_CRS,
             family="serif", fontweight="bold")
    ax4.text(160, 1.05, "Phase III:\nLock-in",
             ha="center", va="bottom", fontsize=7, color=C_LOCK,
             family="serif", fontweight="bold")

    ax4.set_xlabel("Evolutionary generation", fontsize=11, family="serif")
    ax4.set_ylabel("Normalised value", fontsize=11, family="serif")
    ax4.set_title(
        "(d) Evolutionary Trajectory: Realism → Interface Lock-in",
        fontsize=12, fontweight="bold", family="serif", pad=10)
    ax4.legend(fontsize=8, framealpha=0.9, loc="center right")
    ax4.grid(True, alpha=0.15)
    ax4.set_ylim(-0.05, 1.15)

    # ── Supertitle ──
    fig.suptitle(
        "Neuro-Evolutionary Feedback Loop of Constraint-Induced Perception\n"
        "Interface Utility Supplants Veridical Truth via Evolutionary Selection",
        fontsize=16, fontweight="bold", family="serif", y=0.99,
        color=C_TEXT
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("manuscripts/ciir_monograph/figures/fig11_evolutionary_feedback.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig11_evolutionary_feedback.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 11 saved: fig11_evolutionary_feedback.{png,pdf}")


if __name__ == "__main__":
    main()

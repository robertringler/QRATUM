#!/usr/bin/env python3
"""
Figure 3: CRS → CIIR → Observer → CRS Recursive Loop
======================================================
Circular diagram showing the recursive compositional system
and its convergence to quantum mechanics as a fixed point.

Publication-quality, 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Colours ──
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
C_CRS      = "#E67E22"   # orange
C_CIIR     = "#8E44AD"   # purple
C_OBS      = "#2980B9"   # blue
C_FIXED    = "#27AE60"   # green
C_ARROW    = "#34495E"   # dark grey
C_CONV     = "#E74C3C"   # red for convergence


def draw_node(ax, cx, cy, radius, label, sublabel, colour):
    """Draw a circular node."""
    circle = plt.Circle((cx, cy), radius, facecolor=colour,
                         edgecolor=colour, alpha=0.12, linewidth=0, zorder=2)
    ax.add_patch(circle)
    border = plt.Circle((cx, cy), radius, facecolor="none",
                         edgecolor=colour, linewidth=3.0, zorder=3)
    ax.add_patch(border)
    ax.text(cx, cy + 0.02, label, ha="center", va="bottom",
            fontsize=16, fontweight="bold", color=colour,
            family="serif", zorder=4)
    ax.text(cx, cy - 0.04, sublabel, ha="center", va="top",
            fontsize=8, color=C_TEXT, family="serif",
            style="italic", zorder=4)


def draw_curved_arrow(ax, start_angle, end_angle, radius, label,
                      colour=C_ARROW, label_offset=0.12):
    """Draw a curved arrow along a circle."""
    # Generate arc points
    angles = np.linspace(np.radians(start_angle + 10),
                          np.radians(end_angle - 10), 50)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    ax.plot(xs, ys, color=colour, lw=2.5, alpha=0.7, zorder=1)

    # Arrowhead at end
    dx = xs[-1] - xs[-2]
    dy = ys[-1] - ys[-2]
    ax.annotate("", xy=(xs[-1], ys[-1]),
                xytext=(xs[-1] - dx * 3, ys[-1] - dy * 3),
                arrowprops=dict(arrowstyle="-|>", color=colour,
                                lw=2.5, mutation_scale=20),
                zorder=5)

    # Label at midpoint
    mid_angle = np.radians((start_angle + end_angle) / 2)
    lx = (radius + label_offset) * np.cos(mid_angle)
    ly = (radius + label_offset) * np.sin(mid_angle)
    ax.text(lx, ly, label, ha="center", va="center",
            fontsize=9, color=colour, fontweight="bold",
            family="serif", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=2))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Panel A: The Recursive Loop
    # ═══════════════════════════════════════════
    ax = axes[0]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    R = 0.65  # radius of the loop
    node_r = 0.18

    # Three nodes at 120° intervals
    nodes = [
        (90,  C_CRS,  "CRS",  "Computational\nReality\nSubstrate"),
        (210, C_CIIR, "CIIR", "Constraint-\nInduced\nInterface"),
        (330, C_OBS,  "Observer", "Cognitive\nSystem\n$X$"),
    ]

    for angle, col, lbl, sub in nodes:
        cx = R * np.cos(np.radians(angle))
        cy = R * np.sin(np.radians(angle))
        draw_node(ax, cx, cy, node_r, lbl, sub, col)

    # Curved arrows between nodes
    arrows_info = [
        (90, 210,   r"$\mathcal{C}$: Evolution", C_CRS),
        (210, 330,  r"$\Phi$: Interface map", C_CIIR),
        (330, 90,   r"$\mathcal{O}$: Observation", C_OBS),
    ]
    for sa, ea, lbl, col in arrows_info:
        draw_curved_arrow(ax, sa, ea, R, lbl, col, label_offset=0.15)

    # Fixed point at center
    fp_circle = plt.Circle((0, 0), 0.12, facecolor=C_FIXED,
                            alpha=0.15, edgecolor=C_FIXED,
                            linewidth=2.5, zorder=6)
    ax.add_patch(fp_circle)
    ax.text(0, 0.01, r"$\mathbf{QM}$", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=C_FIXED,
            family="serif", zorder=7)
    ax.text(0, -0.03, "Fixed\nPoint", ha="center", va="top",
            fontsize=7, color=C_FIXED, family="serif",
            style="italic", zorder=7)

    ax.set_title("(a) Recursive Compositional Loop",
                 fontsize=14, fontweight="bold", family="serif", pad=15)

    # ═══════════════════════════════════════════
    # Panel B: Convergence to Fixed Point
    # ═══════════════════════════════════════════
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)

    # Simulate convergence: ||T^n(ρ₀) - ρ*||
    n_iter = 30
    iterations = np.arange(n_iter)

    # Trace distance convergence (exponential)
    gamma = 0.7  # contractivity parameter
    trace_dist = 1.0 * gamma**iterations

    # Entropy convergence
    S_max = np.log(4)  # ln(d) for d=4
    entropy = S_max * (1 - 0.8 * gamma**iterations)

    # Plot trace distance
    ax2.semilogy(iterations, trace_dist, "o-", color=C_CONV,
                 lw=2.0, markersize=4, label=r"$\|\rho_n - \rho^*\|_1$",
                 zorder=3)

    # Contractivity bound
    ax2.semilogy(iterations, gamma**iterations, "--",
                 color=C_ARROW, lw=1.5, alpha=0.5,
                 label=r"$\gamma^n$ bound ($\gamma = 0.7$)")

    ax2.set_xlabel("Iteration $n$", fontsize=12, family="serif")
    ax2.set_ylabel("Trace distance to fixed point", fontsize=12,
                   family="serif")
    ax2.set_title("(b) Convergence: $T^n(\\rho_0) \\to \\rho^*$",
                  fontsize=14, fontweight="bold", family="serif", pad=15)
    ax2.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, n_iter - 0.5)

    # Add convergence theorem box
    ax2.text(15, 0.3,
             r"$\|T^n(\rho_0) - \rho^*\|_1 \leq \gamma^n \|\rho_0 - \rho^*\|_1$"
             "\n\n"
             r"$\gamma = \sup_{\rho \neq \rho'} "
             r"\frac{\|T(\rho) - T(\rho')\|_1}{\|\rho - \rho'\|_1} < 1$",
             ha="center", va="center", fontsize=9,
             color=C_TEXT, family="serif",
             bbox=dict(facecolor="white", edgecolor=C_FIXED,
                       boxstyle="round,pad=0.5", linewidth=1.5,
                       alpha=0.95))

    # Indicate fixed point properties
    ax2.text(25, 0.001,
             "Fixed point:\n"
             "• Hilbert space $\\mathcal{H}$\n"
             "• Born rule $p_k = \\mathrm{Tr}(\\rho \\Pi_k)$\n"
             "• Lindblad dynamics",
             ha="center", va="center", fontsize=8,
             color=C_FIXED, family="serif",
             bbox=dict(facecolor=C_FIXED, edgecolor=C_FIXED,
                       boxstyle="round,pad=0.4", alpha=0.1))

    # ── Supertitle ──
    fig.suptitle(
        "Quantum Mechanics as Fixed Point of the CRS–CIIR–Observer Loop",
        fontsize=16, fontweight="bold", family="serif", y=1.0,
        color=C_TEXT
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("manuscripts/ciir_monograph/figures/fig03_recursive_loop.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig03_recursive_loop.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 3 saved: fig03_recursive_loop.{png,pdf}")


if __name__ == "__main__":
    main()

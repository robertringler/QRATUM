#!/usr/bin/env python3
"""
Figure 6: Emergent Spacetime and Gravitational Field Equations
===============================================================
Shows the derivation chain:
  Fisher metric → Christoffel → Riemann → Einstein equations

Includes the Einstein-Hilbert action from constraint geometry
and the field equation derivation.

Publication-quality, 300 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

# ── Colours ──
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
C_FISHER   = "#E74C3C"
C_EINSTEIN = "#2C3E50"
C_CURV     = "#8E44AD"
C_MATTER   = "#E67E22"
C_COSMO    = "#27AE60"
C_ARROW    = "#7F8C8D"


def equation_box(ax, x, y, w, h, title, eqn, colour, fontsize=10):
    """Draw a highlighted equation box."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01",
        facecolor=colour, edgecolor=colour,
        alpha=0.08, linewidth=0, zorder=2
    )
    ax.add_patch(box)
    border = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01",
        facecolor="none", edgecolor=colour,
        linewidth=2.0, zorder=3
    )
    ax.add_patch(border)
    ax.text(x, y + h/2 - 0.015, title,
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=colour, family="serif", zorder=4)
    ax.text(x, y - 0.01, eqn,
            ha="center", va="center", fontsize=fontsize, color=C_TEXT,
            family="serif", zorder=4)


def main():
    fig = plt.figure(figsize=(16, 13), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Top: Derivation flow chart
    # ═══════════════════════════════════════════
    ax1 = fig.add_axes([0.02, 0.62, 0.96, 0.35])
    ax1.set_facecolor(C_BG)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axis("off")

    ax1.text(0.5, 1.0,
             "Emergent Gravitational Field Equations from Constraint Geometry",
             ha="center", va="top", fontsize=15, fontweight="bold",
             color=C_TEXT, family="serif")

    # Equation boxes
    boxes = [
        (0.12, 0.65, 0.20, 0.22,
         "Fisher–Rao Metric",
         r"$g_{ij}^{\mathrm{FR}} = \mathrm{Tr}(\rho\, \partial_i \ln\rho\,"
         r" \partial_j \ln\rho)$",
         C_FISHER),
        (0.38, 0.65, 0.20, 0.22,
         "Einstein–Hilbert Action",
         r"$\mathcal{A}_\mathrm{grav} = \frac{1}{16\pi G_\mathrm{eff}}$"
         "\n"
         r"$\int (R - 2\Lambda_\mathrm{eff}) \sqrt{|g|}\, d^n\sigma$",
         C_CURV),
        (0.64, 0.65, 0.20, 0.22,
         "Variation $\\delta \\mathcal{A} = 0$",
         r"$\frac{\delta \mathcal{A}_\mathrm{grav}}{\delta g^{\mu\nu}} = 0$"
         "\n"
         r"$\Rightarrow$ field equations",
         C_EINSTEIN),
        (0.88, 0.65, 0.20, 0.22,
         "Einstein Equations",
         r"$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R$"
         "\n"
         r"$+ \Lambda_\mathrm{eff}\, g_{\mu\nu} = 8\pi G_\mathrm{eff}\, T_{\mu\nu}$",
         C_COSMO, 9),
    ]

    for args in boxes:
        equation_box(ax1, *args)

    # Arrows
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2] / 2 + 0.01
        x1 = boxes[i+1][0] - boxes[i+1][2] / 2 - 0.01
        y = boxes[i][1]
        ax1.annotate(
            "", xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=2.0, mutation_scale=16)
        )

    # Source terms
    ax1.text(0.88, 0.35,
             r"$T_{\mu\nu} = \nabla_\mu \mathcal{C} \nabla_\nu \mathcal{C}$"
             "\n"
             r"$- \frac{1}{2}g_{\mu\nu} (\nabla \mathcal{C})^2$"
             "\n"
             r"$- g_{\mu\nu} V(\mathcal{C})$",
             ha="center", va="center", fontsize=9, color=C_MATTER,
             family="serif",
             bbox=dict(facecolor=C_MATTER, edgecolor=C_MATTER,
                       alpha=0.08, boxstyle="round,pad=0.3"))
    ax1.text(0.88, 0.22, "Constraint Stress-Energy",
             ha="center", va="top", fontsize=8, color=C_MATTER,
             fontweight="bold", family="serif")

    # Emergent coupling constants
    ax1.text(0.12, 0.35,
             r"$G_\mathrm{eff} = \frac{\ell_0^{n-2}}{8\pi \tilde{\kappa}_\mathrm{max}}$"
             "\n\n"
             r"$\Lambda_\mathrm{eff} = \frac{n(n-1)}{2}\,\kappa_\mathrm{bg}$",
             ha="center", va="center", fontsize=9, color=C_FISHER,
             family="serif",
             bbox=dict(facecolor=C_FISHER, edgecolor=C_FISHER,
                       alpha=0.08, boxstyle="round,pad=0.3"))
    ax1.text(0.12, 0.20, "Emergent Constants",
             ha="center", va="top", fontsize=8, color=C_FISHER,
             fontweight="bold", family="serif")

    # Key insight box
    ax1.text(0.50, 0.12,
             "KEY: No spacetime assumed a priori.\n"
             "Metric $g_{\\mu\\nu}$ emerges from divergence $\\mathcal{D}$,\n"
             "curvature from constraint landscape $\\mathcal{C}$,\n"
             "gravity from variational principle on emergent geometry.",
             ha="center", va="center", fontsize=9, color=C_EINSTEIN,
             family="serif",
             bbox=dict(facecolor="#ECF0F1", edgecolor=C_EINSTEIN,
                       boxstyle="round,pad=0.4", linewidth=2))

    # ═══════════════════════════════════════════
    # Bottom left: Curved spacetime from constraint
    # ═══════════════════════════════════════════
    ax2 = fig.add_subplot(223, projection="3d")
    ax2.set_facecolor(C_BG)

    # Create a curved 2D surface representing spacetime
    u = np.linspace(-np.pi, np.pi, 80)
    v = np.linspace(-np.pi, np.pi, 80)
    U, V = np.meshgrid(u, v)

    # Warped surface (gravitational well)
    mass_x, mass_y = 0, 0
    r = np.sqrt((U - mass_x)**2 + (V - mass_y)**2) + 0.3
    W = -1.5 / r + 0.1 * (U**2 + V**2)

    ax2.plot_surface(U, V, W, cmap="coolwarm", alpha=0.5,
                     edgecolor="none", antialiased=True)

    # Geodesics around the mass
    for phi0 in [0, np.pi/3, 2*np.pi/3, np.pi]:
        t = np.linspace(0, 2*np.pi, 200)
        r_orbit = 1.5 + 0.3 * np.cos(2*t + phi0)
        gx = r_orbit * np.cos(t)
        gy = r_orbit * np.sin(t)
        r_g = np.sqrt(gx**2 + gy**2) + 0.3
        gz = -1.5 / r_g + 0.1 * (gx**2 + gy**2) + 0.02
        ax2.plot(gx, gy, gz, color=C_COSMO, lw=1.5, alpha=0.7)

    ax2.set_xlabel(r"$\sigma^1$", fontsize=10, family="serif")
    ax2.set_ylabel(r"$\sigma^2$", fontsize=10, family="serif")
    ax2.set_title("(b) Emergent Curved Spacetime\n(gravitational well from $\\mathcal{C}$)",
                  fontsize=11, fontweight="bold", family="serif", pad=5)
    ax2.view_init(elev=30, azim=-40)

    # ═══════════════════════════════════════════
    # Bottom right: Limit recovery diagram
    # ═══════════════════════════════════════════
    ax3 = fig.add_subplot(224)
    ax3.set_facecolor(C_BG)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.axis("off")

    # Central CIIR box
    bw, bh = 0.35, 0.15
    equation_box(ax3, 0.5, 0.75, bw, bh,
                 "CIIR Emergent Framework",
                 r"$(\mathcal{S}, \mathcal{C}, \mathcal{D})$",
                 C_CURV)

    # Limit boxes
    limits = [
        (0.15, 0.35, "General Relativity",
         r"$n \to 4,\; \beta \to \infty$",
         C_EINSTEIN),
        (0.50, 0.35, "Quantum Mechanics",
         r"$\mathcal{C} \to 0,\; \gamma < 1$",
         C_COSMO),
        (0.85, 0.35, "Statistical Mechanics",
         r"$\beta$ finite, $N \to \infty$",
         C_MATTER),
        (0.15, 0.08, "Newton's Gravity",
         r"$v/c \to 0,\; R \to 0$",
         "#7F8C8D"),
        (0.85, 0.08, "Classical Mechanics",
         r"$\hbar_\mathrm{eff} \to 0$",
         "#7F8C8D"),
    ]

    for lx, ly, title, cond, col in limits:
        equation_box(ax3, lx, ly, 0.25, 0.12, title, cond, col, fontsize=8)

    # Arrows from CIIR to limits
    for lx, ly, _, _, _ in limits[:3]:
        ax3.annotate(
            "", xy=(lx, ly + 0.06), xytext=(0.5, 0.68),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=1.5, mutation_scale=14,
                            connectionstyle="arc3,rad=0.1")
        )

    # Sub-limits
    ax3.annotate("", xy=(0.15, 0.14), xytext=(0.15, 0.29),
                 arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                 lw=1.2, mutation_scale=12))
    ax3.annotate("", xy=(0.85, 0.14), xytext=(0.50, 0.29),
                 arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                 lw=1.2, mutation_scale=12))

    ax3.set_title("(c) Limit Recovery Structure",
                  fontsize=12, fontweight="bold", family="serif", pad=10)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("manuscripts/ciir_monograph/figures/fig06_emergent_gravity.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig06_emergent_gravity.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 6 saved: fig06_emergent_gravity.{png,pdf}")


if __name__ == "__main__":
    main()

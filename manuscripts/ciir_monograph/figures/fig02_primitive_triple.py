#!/usr/bin/env python3
"""
Figure 2: Primitive Triple (S, C, D)
=====================================
Shows the state space S with:
  - Constraint functional C as a 3D landscape
  - Divergence D as connections between states
  - Constraint-satisfying surface C⁻¹(0)

Publication-quality, 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Colours ──
C_BG = "#FAFAFA"
C_TEXT = "#1A1A2E"
C_SURFACE = "#8E44AD"
C_ZERO = "#27AE60"
C_DIVERG = "#E74C3C"
C_STATE = "#2980B9"


def main():
    fig = plt.figure(figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ── Panel A: Constraint landscape C(σ) ──
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_facecolor(C_BG)

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    # Constraint functional: double-well with saddle
    Z = (X**2 + Y**2 - 1) ** 2 + 0.3 * X * Y

    ax1.plot_surface(X, Y, Z, cmap="magma_r", alpha=0.7, edgecolor="none", antialiased=True)
    # Mark C = 0 contour
    ax1.contour(X, Y, Z, levels=[0.02], zdir="z", offset=0, colors=[C_ZERO], linewidths=2.5)

    ax1.set_xlabel(r"$\sigma^1$", fontsize=11, labelpad=5, family="serif")
    ax1.set_ylabel(r"$\sigma^2$", fontsize=11, labelpad=5, family="serif")
    ax1.set_zlabel(r"$\mathcal{C}(\sigma)$", fontsize=11, labelpad=5, family="serif")
    ax1.set_title(
        r"(a) Constraint Functional $\mathcal{C}: \mathcal{S} \to \mathbb{R}_{\geq 0}$",
        fontsize=12,
        fontweight="bold",
        family="serif",
        pad=10,
    )
    ax1.view_init(elev=25, azim=-45)

    # ── Panel B: Divergence structure D(σ₁, σ₂) ──
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(C_BG)

    # Draw state space as a circle with points
    theta = np.linspace(0, 2 * np.pi, 200)
    r_outer = 1.8
    ax2.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), color=C_STATE, lw=1.5, alpha=0.3)

    # Sample states
    np.random.seed(42)
    n_states = 12
    angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    radii = 0.3 + 1.2 * np.random.rand(n_states)
    sx = radii * np.cos(angles)
    sy = radii * np.sin(angles)

    # Draw divergence connections (asymmetric → directed arrows)
    for i in range(n_states):
        for j in range(i + 1, n_states):
            d = np.sqrt((sx[i] - sx[j]) ** 2 + (sy[i] - sy[j]) ** 2)
            if d < 1.8:
                alpha = max(0.1, 1.0 - d / 2.0)
                ax2.annotate(
                    "",
                    xy=(sx[j], sy[j]),
                    xytext=(sx[i], sy[i]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_DIVERG,
                        lw=0.8,
                        alpha=alpha * 0.5,
                        mutation_scale=8,
                    ),
                )

    ax2.scatter(sx, sy, s=60, c=C_STATE, zorder=5, edgecolor="white", linewidth=1.0)

    # Label two states
    ax2.annotate(
        r"$\sigma_1$",
        (sx[0], sy[0]),
        fontsize=11,
        family="serif",
        color=C_TEXT,
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax2.annotate(
        r"$\sigma_2$",
        (sx[3], sy[3]),
        fontsize=11,
        family="serif",
        color=C_TEXT,
        xytext=(10, 10),
        textcoords="offset points",
    )

    # Highlight one divergence
    ax2.annotate(
        "",
        xy=(sx[3], sy[3]),
        xytext=(sx[0], sy[0]),
        arrowprops=dict(arrowstyle="-|>", color=C_DIVERG, lw=2.5, mutation_scale=15),
    )
    mx = (sx[0] + sx[3]) / 2
    my = (sy[0] + sy[3]) / 2
    ax2.text(
        mx + 0.15,
        my + 0.15,
        r"$\mathcal{D}(\sigma_1, \sigma_2)$",
        fontsize=11,
        color=C_DIVERG,
        family="serif",
        fontweight="bold",
    )

    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-2.2, 2.2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(
        r"(b) Divergence $\mathcal{D}: \mathcal{S} \times \mathcal{S} \to \mathbb{R}_{\geq 0}$",
        fontsize=12,
        fontweight="bold",
        family="serif",
        pad=10,
    )

    # ── Panel C: Induced metric tensor ──
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(C_BG)

    # Draw metric ellipses at grid points
    from matplotlib.patches import Ellipse

    grid_x = np.linspace(-1.5, 1.5, 5)
    grid_y = np.linspace(-1.5, 1.5, 5)

    for gx in grid_x:
        for gy in grid_y:
            # Metric tensor varies with position
            r = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx) * 180 / np.pi
            # Anisotropy increases with distance
            a = 0.15 + 0.05 * r
            b = 0.15 - 0.03 * r
            b = max(b, 0.05)
            ell = Ellipse(
                (gx, gy),
                2 * a,
                2 * b,
                angle=angle + 30,
                facecolor=C_SURFACE,
                alpha=0.15,
                edgecolor=C_SURFACE,
                linewidth=1.2,
            )
            ax3.add_patch(ell)

    ax3.set_xlim(-2.2, 2.2)
    ax3.set_ylim(-2.2, 2.2)
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"$\sigma^1$", fontsize=11, family="serif")
    ax3.set_ylabel(r"$\sigma^2$", fontsize=11, family="serif")
    ax3.set_title(
        r"(c) Induced Metric $g_{ij}(\sigma) = \partial^2 \mathcal{D} / \partial d\sigma^i \partial d\sigma^j$",
        fontsize=12,
        fontweight="bold",
        family="serif",
        pad=10,
    )

    # Add formula
    ax3.text(
        0,
        -1.9,
        r"$\mathcal{D}(\sigma, \sigma + d\sigma) = \frac{1}{2} g_{ij}(\sigma)\, d\sigma^i d\sigma^j + O(\|d\sigma\|^3)$",
        ha="center",
        va="center",
        fontsize=10,
        color=C_TEXT,
        family="serif",
        bbox=dict(facecolor="white", edgecolor=C_SURFACE, boxstyle="round,pad=0.3", alpha=0.9),
    )

    # ── Panel D: Constraint zero set ──
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(C_BG)

    # Contour plot of constraint
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2 - 1) ** 2 + 0.3 * X * Y

    levels = np.linspace(0, 3, 15)
    cs = ax4.contourf(X, Y, Z, levels=levels, cmap="magma_r", alpha=0.6)
    ax4.contour(X, Y, Z, levels=[0.05], colors=[C_ZERO], linewidths=3)

    # Mark minima
    from scipy.optimize import minimize

    for x0 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        res = minimize(
            lambda p: (p[0] ** 2 + p[1] ** 2 - 1) ** 2 + 0.3 * p[0] * p[1], x0, method="Nelder-Mead"
        )
        if res.fun < 0.1:
            ax4.plot(res.x[0], res.x[1], "*", color=C_ZERO, markersize=12, zorder=5)

    ax4.set_xlabel(r"$\sigma^1$", fontsize=11, family="serif")
    ax4.set_ylabel(r"$\sigma^2$", fontsize=11, family="serif")
    ax4.set_title(
        r"(d) Constraint Surface $\mathcal{C}^{-1}(0)$",
        fontsize=12,
        fontweight="bold",
        family="serif",
        pad=10,
    )
    ax4.set_aspect("equal")

    cb = plt.colorbar(cs, ax=ax4, shrink=0.7, label=r"$\mathcal{C}(\sigma)$")
    cb.ax.tick_params(labelsize=8)

    # ── Supertitle ──
    fig.suptitle(
        r"The Primitive Triple $(\mathcal{S},\, \mathcal{C},\, \mathcal{D})$",
        fontsize=16,
        fontweight="bold",
        family="serif",
        y=0.98,
        color=C_TEXT,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig02_primitive_triple.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig02_primitive_triple.pdf",
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.close()
    print("✓ Figure 2 saved: fig02_primitive_triple.{png,pdf}")


if __name__ == "__main__":
    main()

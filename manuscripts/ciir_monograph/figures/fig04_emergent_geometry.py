#!/usr/bin/env python3
"""
Figure 4: Emergent Metric Geometry Pipeline
=============================================
Shows the chain:
  Divergence D → Symmetrised metric d_D → Riemannian manifold (M, g)
  → Christoffel Γ → Riemann R → Ricci R_{ij} → Scalar R

Publication-quality, 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Colours ──
C_BG = "#FAFAFA"
C_TEXT = "#1A1A2E"
C_DIV = "#E74C3C"  # divergence
C_METRIC = "#E67E22"  # metric
C_MANIFOLD = "#8E44AD"  # manifold
C_CONN = "#2980B9"  # connection
C_CURV = "#27AE60"  # curvature
C_ARROW = "#7F8C8D"


def pipeline_box(ax, x, y, w, h, title, formula, colour):
    """Draw a pipeline stage box."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.01",
        facecolor=colour,
        edgecolor=colour,
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    ax.add_patch(box)
    border = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.01",
        facecolor="none",
        edgecolor=colour,
        linewidth=2.5,
        zorder=3,
    )
    ax.add_patch(border)
    ax.text(
        x,
        y + h / 2 - 0.03,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=colour,
        family="serif",
        zorder=4,
    )
    ax.text(
        x,
        y - 0.01,
        formula,
        ha="center",
        va="center",
        fontsize=9,
        color=C_TEXT,
        family="serif",
        zorder=4,
    )


def main():
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Top panel: Pipeline diagram
    # ═══════════════════════════════════════════
    ax1 = fig.add_axes([0.02, 0.55, 0.96, 0.40])
    ax1.set_facecolor(C_BG)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.1, 1.0)
    ax1.axis("off")

    ax1.text(
        0.5,
        0.95,
        "Emergence Pipeline: Divergence → Spacetime Geometry",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=C_TEXT,
        family="serif",
    )

    # Pipeline stages
    stages = [
        (
            0.08,
            0.50,
            0.14,
            0.30,
            "Divergence",
            r"$\mathcal{D}(\sigma_1, \sigma_2)$" "\n" r"(asymmetric)",
            C_DIV,
        ),
        (
            0.27,
            0.50,
            0.14,
            0.30,
            "Metric",
            r"$d_\mathcal{D} = \sqrt{\mathcal{D} + \mathcal{D}^T}$" "\n" r"(symmetric)",
            C_METRIC,
        ),
        (
            0.46,
            0.50,
            0.14,
            0.30,
            "Manifold",
            r"$(\mathcal{M}_\mathrm{eff}, g)$" "\n" r"$g_{ij} = \partial^2 \mathcal{D}$",
            C_MANIFOLD,
        ),
        (
            0.65,
            0.50,
            0.14,
            0.30,
            "Connection",
            r"$\Gamma^k_{ij} = \frac{1}{2}g^{k\ell}$" "\n" r"$(\partial_i g_{j\ell} + \ldots)$",
            C_CONN,
        ),
        (0.84, 0.50, 0.14, 0.30, "Curvature", r"$R^k{}_{lij}$" "\n" r"$R_{ij},\; R$", C_CURV),
    ]

    for x, y, w, h, title, formula, col in stages:
        pipeline_box(ax1, x, y, w, h, title, formula, col)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x0 = stages[i][0] + stages[i][2] / 2 + 0.01
        x1 = stages[i + 1][0] - stages[i + 1][2] / 2 - 0.01
        y = stages[i][1]
        ax1.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=2.5, mutation_scale=18),
        )

    # Labels below
    labels = [
        (0.08, "Def. 14.1"),
        (0.27, "Thm. 14.1"),
        (0.46, "Thm. 14.3"),
        (0.65, "Def. 14.4"),
        (0.84, "Thm. 14.4"),
    ]
    for x, lbl in labels:
        ax1.text(
            x,
            0.28,
            lbl,
            ha="center",
            va="top",
            fontsize=8,
            color=C_ARROW,
            family="serif",
            style="italic",
        )

    # ═══════════════════════════════════════════
    # Bottom left: Geodesic on curved manifold
    # ═══════════════════════════════════════════
    ax2 = fig.add_subplot(223, projection="3d")
    ax2.set_facecolor(C_BG)

    # Curved surface
    u = np.linspace(-2, 2, 80)
    v = np.linspace(-2, 2, 80)
    U, V = np.meshgrid(u, v)
    W = 0.3 * np.sin(U) * np.cos(V) + 0.1 * U

    ax2.plot_surface(U, V, W, cmap="coolwarm", alpha=0.4, edgecolor="none", antialiased=True)

    # Geodesic curve
    t = np.linspace(-1.5, 1.5, 100)
    gx = t
    gy = 0.5 * np.sin(t * 1.5)
    gz = 0.3 * np.sin(gx) * np.cos(gy) + 0.1 * gx + 0.01
    ax2.plot(gx, gy, gz, color=C_CURV, lw=3.0, zorder=5)
    ax2.scatter([gx[0]], [gy[0]], [gz[0]], color=C_DIV, s=80, zorder=6)
    ax2.scatter([gx[-1]], [gy[-1]], [gz[-1]], color=C_CONN, s=80, zorder=6)

    # Tangent vectors at a few points
    for idx in [20, 50, 80]:
        dx = gx[idx + 1] - gx[idx - 1]
        dy = gy[idx + 1] - gy[idx - 1]
        dz = gz[idx + 1] - gz[idx - 1]
        scale = 0.3
        ax2.quiver(
            gx[idx],
            gy[idx],
            gz[idx],
            dx * scale,
            dy * scale,
            dz * scale,
            color=C_MANIFOLD,
            arrow_length_ratio=0.3,
            linewidth=1.5,
        )

    ax2.set_xlabel(r"$\sigma^1$", fontsize=10, family="serif")
    ax2.set_ylabel(r"$\sigma^2$", fontsize=10, family="serif")
    ax2.set_zlabel(r"$\mathcal{C}$", fontsize=10, family="serif")
    ax2.set_title(
        "(b) Geodesic on Emergent Manifold", fontsize=12, fontweight="bold", family="serif", pad=10
    )
    ax2.view_init(elev=20, azim=-50)

    # ═══════════════════════════════════════════
    # Bottom right: Curvature visualization
    # ═══════════════════════════════════════════
    ax3 = fig.add_subplot(224)
    ax3.set_facecolor(C_BG)

    # Gaussian curvature heat map
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)

    # Scalar curvature from constraint functional
    C_func = (X**2 + Y**2 - 1) ** 2 + 0.3 * X * Y
    # Approximate curvature as Laplacian of log-metric-determinant
    from scipy.ndimage import laplace

    R_approx = -laplace(np.log(1 + C_func + 0.01))

    im = ax3.imshow(
        R_approx,
        extent=[-2, 2, -2, 2],
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-5,
        vmax=5,
    )
    ax3.contour(X, Y, C_func, levels=[0.05], colors=[C_CURV], linewidths=2)

    ax3.set_xlabel(r"$\sigma^1$", fontsize=11, family="serif")
    ax3.set_ylabel(r"$\sigma^2$", fontsize=11, family="serif")
    ax3.set_title(
        "(c) Scalar Curvature $R(\\sigma)$", fontsize=12, fontweight="bold", family="serif", pad=10
    )
    cb = plt.colorbar(im, ax=ax3, shrink=0.7, label=r"$R = g^{ij} R_{ij}$")
    cb.ax.tick_params(labelsize=8)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig04_emergent_geometry.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig04_emergent_geometry.pdf",
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.close()
    print("✓ Figure 4 saved: fig04_emergent_geometry.{png,pdf}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 10: Objective Complexity vs. Interface Icons
=====================================================
GPAI STEM VISUALIZER — Prompt:
  "CIIR interface theory: comparative schematic of objective ontic
   complexity (high-dimensional, densely interconnected, continuous)
   versus simplified epistemic 'desktop icons' (discrete, finite,
   user-friendly). Show the constraint operator Ĉ as the mapping
   that compresses reality into functional representations.
   Inspired by Hoffman's Interface Theory of Perception (ITP).
   Scientific Diagram Style."

Design choices:
  - Left panel: dense, high-dimensional complexity cloud (ontic)
  - Right panel: sparse, discrete, coloured icons (epistemic)
  - Central panel: constraint bottleneck with information metrics
  - Bottom: quantitative mutual information comparison
  - Nature/Science journal colour palette, 300 DPI

Publication-quality, 4K-equivalent at 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

# ── Colour palette ──
C_BG = "#FAFAFA"
C_TEXT = "#1A1A2E"
C_ONTIC = "#2C3E50"
C_CONSTR = "#8E44AD"
C_IFACE = "#E74C3C"
C_EPISTEMIC = "#2980B9"
C_ICON = ["#E74C3C", "#27AE60", "#F39C12", "#2980B9", "#8E44AD", "#1ABC9C", "#E67E22", "#2C3E50"]
C_ARROW = "#7F8C8D"


def draw_complexity_cloud(ax, cx, cy, radius, n_points=800, n_edges=400):
    """Draw a dense complexity cloud representing ontic structure."""
    np.random.seed(17)
    # Random points in a cluster
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    px = cx + radii * np.cos(angles)
    py = cy + radii * np.sin(angles)

    # Draw dense edges (interconnections)
    for _ in range(n_edges):
        i, j = np.random.randint(0, n_points, 2)
        d = np.sqrt((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2)
        if d < radius * 0.6:
            ax.plot([px[i], px[j]], [py[i], py[j]], color=C_ONTIC, lw=0.15, alpha=0.12)

    # Draw points with varying sizes (representing degrees of freedom)
    sizes = 2 + 8 * np.random.rand(n_points)
    alphas = 0.15 + 0.3 * np.random.rand(n_points)
    for i in range(n_points):
        ax.plot(px[i], py[i], "o", color=C_ONTIC, markersize=sizes[i] * 0.3, alpha=alphas[i])

    return px, py


def draw_interface_icons(ax, cx, cy, spacing=0.22):
    """Draw discrete interface icons (desktop metaphor)."""
    icons = [
        (cx - spacing * 1.5, cy + spacing, "◆", C_ICON[0], r"$A_1$"),
        (cx - spacing * 0.5, cy + spacing, "●", C_ICON[1], r"$A_2$"),
        (cx + spacing * 0.5, cy + spacing, "■", C_ICON[2], r"$A_3$"),
        (cx + spacing * 1.5, cy + spacing, "▲", C_ICON[3], r"$A_4$"),
        (cx - spacing, cy - spacing * 0.2, "★", C_ICON[4], r"$A_5$"),
        (cx, cy - spacing * 0.2, "⬟", C_ICON[5], r"$A_6$"),
        (cx + spacing, cy - spacing * 0.2, "◯", C_ICON[6], r"$A_7$"),
    ]

    for ix, iy, shape, col, label in icons:
        # Icon background
        box = FancyBboxPatch(
            (ix - 0.08, iy - 0.08),
            0.16,
            0.16,
            boxstyle="round,pad=0.02",
            facecolor=col,
            edgecolor=col,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        ax.add_patch(box)
        border = FancyBboxPatch(
            (ix - 0.08, iy - 0.08),
            0.16,
            0.16,
            boxstyle="round,pad=0.02",
            facecolor="none",
            edgecolor=col,
            linewidth=2.0,
            zorder=3,
        )
        ax.add_patch(border)

        # Icon symbol
        ax.text(ix, iy + 0.01, shape, ha="center", va="center", fontsize=16, color=col, zorder=4)
        # Label
        ax.text(
            ix,
            iy - 0.10,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=col,
            family="serif",
            zorder=4,
        )


def main():
    fig = plt.figure(figsize=(20, 14), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════════════════════════
    # Main panel: three-column layout
    # ═══════════════════════════════════════════════════════════════
    ax_main = fig.add_axes([0.02, 0.35, 0.96, 0.58])
    ax_main.set_facecolor(C_BG)
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(-0.15, 1.05)
    ax_main.axis("off")

    # Title
    ax_main.text(
        0.5,
        1.03,
        "Objective Complexity vs. Interface Icons",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=C_TEXT,
        family="serif",
    )
    ax_main.text(
        0.5,
        0.97,
        r"The Constraint Operator $\hat{C}$ Compresses "
        r"$\infty$-Dimensional Reality into Finite Epistemic Representations",
        ha="center",
        va="top",
        fontsize=10,
        color=C_ARROW,
        family="serif",
        style="italic",
    )

    # ── Left: Objective Complexity ──
    ax_main.text(
        0.17,
        0.88,
        r"ONTIC: $\mathcal{R}$",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=C_ONTIC,
        family="serif",
    )
    ax_main.text(
        0.17,
        0.84,
        '"Ding an sich"',
        ha="center",
        va="bottom",
        fontsize=9,
        color=C_ONTIC,
        family="serif",
        style="italic",
    )

    draw_complexity_cloud(ax_main, 0.17, 0.48, 0.14)

    # Properties below
    ax_main.text(
        0.17,
        0.28,
        "• Continuous\n• $\\infty$-dimensional\n"
        "• Densely interconnected\n• Non-separable structure\n"
        "• Inaccessible directly",
        ha="center",
        va="top",
        fontsize=8,
        color=C_ONTIC,
        family="serif",
        linespacing=1.6,
    )

    # ── Center: Constraint bottleneck ──
    # Funnel shape
    funnel_x_top = [0.33, 0.67]
    funnel_x_bot = [0.43, 0.57]
    funnel_y_top = 0.80
    funnel_y_bot = 0.20

    # Draw funnel edges
    ax_main.plot(
        [funnel_x_top[0], funnel_x_bot[0]],
        [funnel_y_top, funnel_y_bot],
        color=C_CONSTR,
        lw=2.5,
        alpha=0.6,
    )
    ax_main.plot(
        [funnel_x_top[1], funnel_x_bot[1]],
        [funnel_y_top, funnel_y_bot],
        color=C_CONSTR,
        lw=2.5,
        alpha=0.6,
    )

    # Fill funnel
    from matplotlib.patches import Polygon as MplPolygon

    funnel = MplPolygon(
        [
            [funnel_x_top[0], funnel_y_top],
            [funnel_x_top[1], funnel_y_top],
            [funnel_x_bot[1], funnel_y_bot],
            [funnel_x_bot[0], funnel_y_bot],
        ],
        facecolor=C_CONSTR,
        alpha=0.06,
        edgecolor="none",
        zorder=0,
    )
    ax_main.add_patch(funnel)

    # Label
    ax_main.text(
        0.50,
        0.88,
        "CONSTRAINT BOTTLENECK",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=C_CONSTR,
        family="serif",
    )
    ax_main.text(
        0.50,
        0.52,
        r"$\hat{C} \in \Gamma(\mathrm{End}(T\mathcal{M}))$",
        ha="center",
        va="center",
        fontsize=11,
        color=C_CONSTR,
        family="serif",
        bbox=dict(facecolor="white", edgecolor=C_CONSTR, boxstyle="round,pad=0.3", alpha=0.9),
    )

    # CPTP channel label
    ax_main.text(
        0.50,
        0.42,
        r"$\Phi$: CPTP channel" "\n" r"$\Phi(\rho) = \sum_k K_k \rho K_k^\dagger$",
        ha="center",
        va="center",
        fontsize=9,
        color=C_IFACE,
        family="serif",
        bbox=dict(facecolor="white", edgecolor=C_IFACE, boxstyle="round,pad=0.3", alpha=0.9),
    )

    # Information metrics
    ax_main.text(
        0.50,
        0.30,
        r"$I(\mathcal{R} ; \mathcal{H}) \ll H(\mathcal{R})$"
        "\n"
        r"$S(\Phi(\rho)) \geq S(\rho)$"
        "\n(entropy non-decrease)",
        ha="center",
        va="center",
        fontsize=8,
        color=C_TEXT,
        family="serif",
        bbox=dict(facecolor="#ECF0F1", edgecolor=C_ARROW, boxstyle="round,pad=0.3", alpha=0.9),
    )

    # Arrows through funnel
    # Many arrows entering, few leaving
    for i in range(15):
        y_start = 0.75 - i * 0.003
        x_offset = (i - 7) * 0.015
        ax_main.annotate(
            "",
            xy=(0.42 + x_offset * 0.3, 0.25),
            xytext=(0.35 + x_offset, y_start),
            arrowprops=dict(arrowstyle="-", color=C_ONTIC, lw=0.5, alpha=0.15),
        )

    # ── Right: Interface Icons (Epistemic) ──
    ax_main.text(
        0.83,
        0.88,
        r"EPISTEMIC: $\mathcal{A}(\mathcal{H})$",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=C_EPISTEMIC,
        family="serif",
    )
    ax_main.text(
        0.83,
        0.84,
        '"Desktop Icons"',
        ha="center",
        va="bottom",
        fontsize=9,
        color=C_EPISTEMIC,
        family="serif",
        style="italic",
    )

    draw_interface_icons(ax_main, 0.83, 0.55)

    # Properties
    ax_main.text(
        0.83,
        0.28,
        "• Discrete\n• Finite-dimensional\n"
        "• Sparse, separable\n• User-friendly structure\n"
        "• Operationally accessible",
        ha="center",
        va="top",
        fontsize=8,
        color=C_EPISTEMIC,
        family="serif",
        linespacing=1.6,
    )

    # Big arrows left→center→right
    ax_main.annotate(
        "",
        xy=(0.33, 0.50),
        xytext=(0.28, 0.50),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=3.0, mutation_scale=20),
    )
    ax_main.annotate(
        "",
        xy=(0.72, 0.50),
        xytext=(0.67, 0.50),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=3.0, mutation_scale=20),
    )

    # ═══════════════════════════════════════════════════════════════
    # Bottom panel: Quantitative information comparison
    # ═══════════════════════════════════════════════════════════════
    ax_bar = fig.add_axes([0.08, 0.05, 0.40, 0.25])
    ax_bar.set_facecolor(C_BG)

    categories = [
        "Spatial\ndimensions",
        "Spectral\nresolution",
        "Temporal\ngranularity",
        "Phase space\nvolume",
        "Causal\nstructure",
    ]
    ontic_vals = [3e10, 1e14, 1e43, 1e120, 1e60]
    epistemic_vals = [3, 1e6, 1e2, 1e10, 1e3]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax_bar.bar(
        x - width / 2,
        [np.log10(v) for v in ontic_vals],
        width,
        color=C_ONTIC,
        alpha=0.7,
        label=r"$\mathcal{R}$ (ontic)",
    )
    bars2 = ax_bar.bar(
        x + width / 2,
        [np.log10(v) for v in epistemic_vals],
        width,
        color=C_EPISTEMIC,
        alpha=0.7,
        label=r"$\mathcal{H}$ (epistemic)",
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=7, family="serif")
    ax_bar.set_ylabel(r"$\log_{10}$(degrees of freedom)", fontsize=9, family="serif")
    ax_bar.set_title(
        "(b) Complexity Compression Ratio", fontsize=11, fontweight="bold", family="serif"
    )
    ax_bar.legend(fontsize=8, framealpha=0.9)
    ax_bar.grid(True, alpha=0.15, axis="y")

    # ═══════════════════════════════════════════════════════════════
    # Bottom right: Veridical vs Interface perception diagram
    # ═══════════════════════════════════════════════════════════════
    ax_venn = fig.add_axes([0.55, 0.05, 0.40, 0.25])
    ax_venn.set_facecolor(C_BG)
    ax_venn.set_xlim(-0.1, 1.1)
    ax_venn.set_ylim(-0.1, 1.1)
    ax_venn.axis("off")

    # Two overlapping circles
    truth_circle = Circle(
        (0.35, 0.50), 0.30, facecolor=C_ONTIC, alpha=0.08, edgecolor=C_ONTIC, linewidth=2.0
    )
    ax_venn.add_patch(truth_circle)
    ax_venn.text(
        0.20,
        0.50,
        "Ontic\nTruth",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_ONTIC,
        family="serif",
    )

    utility_circle = Circle(
        (0.65, 0.50), 0.30, facecolor=C_EPISTEMIC, alpha=0.08, edgecolor=C_EPISTEMIC, linewidth=2.0
    )
    ax_venn.add_patch(utility_circle)
    ax_venn.text(
        0.80,
        0.50,
        "Fitness\nUtility",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_EPISTEMIC,
        family="serif",
    )

    # Overlap = veridical perception (small)
    ax_venn.text(
        0.50, 0.50, "∩", ha="center", va="center", fontsize=18, color=C_CONSTR, fontweight="bold"
    )

    # Key insight
    ax_venn.text(
        0.50,
        0.05,
        "CIIR: Interface icons optimise fitness,\n" "not veridical truth. Overlap ≈ 0 generically.",
        ha="center",
        va="center",
        fontsize=8,
        color=C_TEXT,
        family="serif",
        style="italic",
        bbox=dict(facecolor="#ECF0F1", edgecolor=C_ARROW, boxstyle="round,pad=0.3", alpha=0.9),
    )

    ax_venn.set_title(
        r"(c) Veridical Perception $\cap$ Interface: $\approx \emptyset$",
        fontsize=11,
        fontweight="bold",
        family="serif",
        pad=5,
    )

    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig10_complexity_vs_icons.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.savefig(
        "manuscripts/ciir_monograph/figures/fig10_complexity_vs_icons.pdf",
        bbox_inches="tight",
        facecolor=C_BG,
    )
    plt.close()
    print("✓ Figure 10 saved: fig10_complexity_vs_icons.{png,pdf}")


if __name__ == "__main__":
    main()

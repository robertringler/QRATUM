#!/usr/bin/env python3
"""
Figure 1: CIIR Architecture Overview
=====================================
Layered diagram showing the full CIIR ontological–epistemic pipeline:

  Reality Space (R) → Constraint Manifold (M) → Interface Map (Φ)
  → Cognitive State Space (H) → Observable Algebra A(H)

Publication-quality, 300 DPI, Nature-style colour palette.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── colour palette (Nature / PRL style) ─────────────────────────
C_ONTIC    = "#2C3E50"   # dark blue-grey  — ontic stratum
C_CONSTR   = "#8E44AD"   # deep purple     — constraint layer
C_IFACE    = "#E74C3C"   # vermillion      — interface
C_EPISTEMIC = "#2980B9"  # steel blue      — epistemic (Hilbert)
C_OBS      = "#27AE60"   # emerald         — observable algebra
C_ARROW    = "#7F8C8D"   # grey            — arrows
C_BG       = "#FAFAFA"   # off-white       — background
C_TEXT     = "#1A1A2E"   # near-black      — text

def make_layer_box(ax, x, y, w, h, label, sublabel, colour, alpha=0.15):
    """Draw a rounded box with label."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=colour, edgecolor=colour,
        alpha=alpha, linewidth=2.0, zorder=2
    )
    ax.add_patch(box)
    border = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor="none", edgecolor=colour,
        linewidth=2.5, zorder=3
    )
    ax.add_patch(border)
    ax.text(x, y + 0.01, label,
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=colour, zorder=4, family="serif")
    ax.text(x, y - 0.02, sublabel,
            ha="center", va="top", fontsize=9, color=C_TEXT,
            zorder=4, family="serif", style="italic")

def draw_arrow(ax, x0, y0, x1, y1, label, colour=C_ARROW):
    """Draw a styled arrow with label."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=colour,
            lw=2.0, mutation_scale=18,
            connectionstyle="arc3,rad=0.0"
        ), zorder=5
    )
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(mx + 0.02, my, label,
            ha="left", va="center", fontsize=9, color=colour,
            fontweight="bold", family="serif", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.8, pad=1))


def main():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.98, "Constraint-Induced Interface Realism (CIIR)",
            ha="center", va="top", fontsize=18, fontweight="bold",
            color=C_TEXT, family="serif")
    ax.text(0.5, 0.93, "Ontological–Epistemic Architecture",
            ha="center", va="top", fontsize=12, color=C_ARROW,
            family="serif", style="italic")

    # ── Layers (left → right) ──
    bw, bh = 0.16, 0.22
    y_main = 0.50

    positions = [
        (0.10, y_main, C_ONTIC,     r"$\mathcal{R}$",
         "Reality Space\n(Polish space)"),
        (0.30, y_main, C_CONSTR,    r"$(\mathcal{M}, g, \hat{C})$",
         "Constraint\nManifold"),
        (0.50, y_main, C_IFACE,     r"$\Phi$",
         "Interface Map\n(CPTP channel)"),
        (0.70, y_main, C_EPISTEMIC, r"$\mathcal{H}$",
         "Cognitive State\nSpace (Hilbert)"),
        (0.90, y_main, C_OBS,       r"$\mathcal{A}(\mathcal{H})$",
         "Observable\nAlgebra"),
    ]

    for x, y, col, lbl, sub in positions:
        make_layer_box(ax, x, y, bw, bh, lbl, sub, col)

    # ── Arrows ──
    gap = 0.005
    arrows = [
        (0.10 + bw/2 + gap, y_main, 0.30 - bw/2 - gap, y_main,
         r"$\iota$"),
        (0.30 + bw/2 + gap, y_main, 0.50 - bw/2 - gap, y_main,
         r"$\Phi$"),
        (0.50 + bw/2 + gap, y_main, 0.70 - bw/2 - gap, y_main,
         r"$\rho$"),
        (0.70 + bw/2 + gap, y_main, 0.90 - bw/2 - gap, y_main,
         r"$\pi$"),
    ]
    for x0, y0, x1, y1, lbl in arrows:
        draw_arrow(ax, x0, y0, x1, y1, lbl)

    # ── Stratum labels ──
    ax.text(0.20, 0.18, "ONTIC STRATUM",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_ONTIC, family="serif", alpha=0.7)
    ax.plot([0.02, 0.38], [0.22, 0.22], color=C_ONTIC,
            lw=1.5, alpha=0.3, zorder=1)

    ax.text(0.80, 0.18, "EPISTEMIC STRATUM",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_EPISTEMIC, family="serif", alpha=0.7)
    ax.plot([0.62, 0.98], [0.22, 0.22], color=C_EPISTEMIC,
            lw=1.5, alpha=0.3, zorder=1)

    ax.text(0.50, 0.18, "INTERFACE",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_IFACE, family="serif", alpha=0.7)

    # ── Key properties ──
    props = [
        (0.10, 0.76, "• Complete separable metric\n• Borel σ-algebra\n"
         "• Independent of observer",
         C_ONTIC),
        (0.30, 0.76, "• Riemannian (M, g)\n• Curvature κ ≥ 0\n"
         "• Constraint ĉ ∈ C∞(M)",
         C_CONSTR),
        (0.50, 0.76, "• Completely positive\n• Trace preserving\n"
         "• Information-limiting",
         C_IFACE),
        (0.70, 0.76, "• Separable Hilbert space\n• Density operators ρ\n"
         "• Born rule emerges",
         C_EPISTEMIC),
        (0.90, 0.76, "• C*-algebra\n• Self-adjoint operators\n"
         "• Spectral theorem",
         C_OBS),
    ]
    for x, y, txt, col in props:
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=7,
                color=col, family="monospace", alpha=0.8,
                linespacing=1.5)

    # ── Axiom labels ──
    ax.text(0.20, 0.33, "Ax.4.1: Ontological Priority",
            ha="center", va="center", fontsize=7, color=C_ONTIC,
            family="serif", style="italic", alpha=0.6)
    ax.text(0.50, 0.33, "Ax.4.3: Interface Constraint",
            ha="center", va="center", fontsize=7, color=C_IFACE,
            family="serif", style="italic", alpha=0.6)
    ax.text(0.80, 0.33, "Ax.4.5: Measurability",
            ha="center", va="center", fontsize=7, color=C_EPISTEMIC,
            family="serif", style="italic", alpha=0.6)

    # ── Footer ──
    ax.text(0.5, 0.02,
            "CIIR Monograph — Chapter 3: Primitive Definitions, "
            "Chapter 4: Axiom System",
            ha="center", va="bottom", fontsize=7, color=C_ARROW,
            family="serif", alpha=0.5)

    plt.tight_layout()
    plt.savefig("manuscripts/ciir_monograph/figures/fig01_ciir_architecture.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig01_ciir_architecture.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 1 saved: fig01_ciir_architecture.{png,pdf}")


if __name__ == "__main__":
    main()

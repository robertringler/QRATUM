#!/usr/bin/env python3
"""
Figure 8: Functorial Structure — CogSys → Hilb_CIIR
=====================================================
Category-theoretic diagram showing:
  (a) The functor F: CogSys → Hilb_CIIR
  (b) Symmetric monoidal structure
  (c) Natural transformations
  (d) The full CIIR categorical hierarchy

Publication-quality, 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Colours ──
C_BG        = "#FAFAFA"
C_TEXT      = "#1A1A2E"
C_COGSYS    = "#E74C3C"   # red
C_HILB      = "#2980B9"   # blue
C_FUNCTOR   = "#8E44AD"   # purple
C_NAT       = "#27AE60"   # green
C_MONOIDAL  = "#E67E22"   # orange
C_ARROW     = "#7F8C8D"


def cat_node(ax, x, y, label, colour, size=0.08):
    """Draw a category/object node."""
    circle = plt.Circle((x, y), size, facecolor=colour,
                         alpha=0.12, edgecolor=colour,
                         linewidth=2.0, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=colour,
            family="serif", zorder=4)


def morphism_arrow(ax, x0, y0, x1, y1, label, colour=C_ARROW,
                   rad=0.0, fontsize=9):
    """Draw a morphism arrow with label."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=colour,
                        lw=2.0, mutation_scale=15,
                        connectionstyle=f"arc3,rad={rad}")
    )
    mx = (x0 + x1) / 2 + rad * 0.3
    my = (y0 + y1) / 2 + abs(rad) * 0.15 * np.sign(rad)
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=fontsize, color=colour, fontweight="bold",
            family="serif",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.9, pad=1))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Panel A: The Representation Functor
    # ═══════════════════════════════════════════
    ax = axes[0, 0]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")

    # CogSys category
    cat_node(ax, 0.2, 0.9, r"$X_1$", C_COGSYS)
    cat_node(ax, 0.2, 0.3, r"$X_2$", C_COGSYS)
    morphism_arrow(ax, 0.2, 0.82, 0.2, 0.38, r"$f$", C_COGSYS)
    ax.text(0.2, 1.15, r"$\mathbf{CogSys}$",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_COGSYS, family="serif")

    # Functor arrow
    ax.annotate(
        "", xy=(0.65, 0.6), xytext=(0.35, 0.6),
        arrowprops=dict(arrowstyle="-|>", color=C_FUNCTOR,
                        lw=3.0, mutation_scale=20)
    )
    ax.text(0.50, 0.67, r"$\mathbf{F}$",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C_FUNCTOR, family="serif")

    # Hilb_CIIR category
    cat_node(ax, 0.8, 0.9, r"$\mathbf{F}(X_1)$", C_HILB, size=0.10)
    cat_node(ax, 0.8, 0.3, r"$\mathbf{F}(X_2)$", C_HILB, size=0.10)
    morphism_arrow(ax, 0.8, 0.80, 0.8, 0.40, r"$\mathbf{F}(f)$", C_HILB)
    ax.text(0.8, 1.15, r"$\mathbf{Hilb_{CIIR}}$",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_HILB, family="serif")

    ax.set_title("(a) Representation Functor $\\mathbf{F}$",
                 fontsize=13, fontweight="bold", family="serif", pad=15)

    # ═══════════════════════════════════════════
    # Panel B: Commutative diagram for F
    # ═══════════════════════════════════════════
    ax = axes[0, 1]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # Objects
    cat_node(ax, 0.15, 0.85, r"$X$", C_COGSYS, 0.07)
    cat_node(ax, 0.85, 0.85, r"$(\mathcal{H}, \rho, \Phi)$",
             C_HILB, 0.12)
    cat_node(ax, 0.15, 0.25, r"$X'$", C_COGSYS, 0.07)
    cat_node(ax, 0.85, 0.25, r"$(\mathcal{H}', \rho', \Phi')$",
             C_HILB, 0.12)

    # Horizontal arrows (functor)
    morphism_arrow(ax, 0.22, 0.85, 0.73, 0.85,
                   r"$\mathbf{F}$", C_FUNCTOR)
    morphism_arrow(ax, 0.22, 0.25, 0.73, 0.25,
                   r"$\mathbf{F}$", C_FUNCTOR)

    # Vertical arrows (morphisms)
    morphism_arrow(ax, 0.15, 0.78, 0.15, 0.32,
                   r"$f$", C_COGSYS)
    morphism_arrow(ax, 0.85, 0.73, 0.85, 0.37,
                   r"$\mathbf{F}(f)$", C_HILB)

    # Commutes label
    ax.text(0.50, 0.55, "commutes",
            ha="center", va="center", fontsize=10, color=C_NAT,
            family="serif", style="italic",
            bbox=dict(facecolor=C_NAT, edgecolor=C_NAT,
                      alpha=0.08, boxstyle="round,pad=0.2"))

    ax.set_title("(b) Functoriality: $\\mathbf{F}(g \\circ f) = \\mathbf{F}(g) \\circ \\mathbf{F}(f)$",
                 fontsize=12, fontweight="bold", family="serif", pad=15)

    # ═══════════════════════════════════════════
    # Panel C: Monoidal Structure
    # ═══════════════════════════════════════════
    ax = axes[1, 0]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # Tensor product diagram
    cat_node(ax, 0.15, 0.80, r"$X_1$", C_COGSYS, 0.06)
    cat_node(ax, 0.15, 0.50, r"$X_2$", C_COGSYS, 0.06)
    cat_node(ax, 0.50, 0.65, r"$X_1 \otimes X_2$", C_MONOIDAL, 0.10)

    morphism_arrow(ax, 0.21, 0.77, 0.40, 0.68, "", C_MONOIDAL)
    morphism_arrow(ax, 0.21, 0.53, 0.40, 0.62, "", C_MONOIDAL)

    # Functor
    morphism_arrow(ax, 0.60, 0.65, 0.75, 0.65, r"$\mathbf{F}$",
                   C_FUNCTOR)

    cat_node(ax, 0.90, 0.65, r"$\mathcal{H}_1 \otimes \mathcal{H}_2$",
             C_HILB, 0.10)

    # Iso arrow
    ax.text(0.50, 0.35,
            r"$\mathbf{F}(X_1 \otimes X_2) \cong "
            r"\mathbf{F}(X_1) \otimes \mathbf{F}(X_2)$",
            ha="center", va="center", fontsize=10, color=C_MONOIDAL,
            family="serif", fontweight="bold",
            bbox=dict(facecolor=C_MONOIDAL, edgecolor=C_MONOIDAL,
                      alpha=0.08, boxstyle="round,pad=0.3"))

    ax.text(0.50, 0.15,
            "Symmetric monoidal functor:\n"
            r"preserves tensor products $\otimes$," "\n"
            r"unit object $\mathbf{1}$, and" "\n"
            "braiding isomorphisms",
            ha="center", va="center", fontsize=9, color=C_TEXT,
            family="serif", style="italic")

    ax.set_title("(c) Symmetric Monoidal Structure",
                 fontsize=13, fontweight="bold", family="serif", pad=15)

    # ═══════════════════════════════════════════
    # Panel D: Full categorical hierarchy
    # ═══════════════════════════════════════════
    ax = axes[1, 1]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    # Hierarchy of categories
    hierarchy = [
        (0.5, 0.90, r"$\mathbf{CogSys}$", C_COGSYS,
         "Cognitive systems + morphisms"),
        (0.5, 0.70, r"$\mathbf{Hilb_{CIIR}}$", C_HILB,
         "Hilbert spaces + CPTP maps"),
        (0.25, 0.50, r"$\mathbf{Hilb}$", "#7F8C8D",
         "Standard Hilbert category"),
        (0.75, 0.50, r"$\mathbf{CPTP}$", C_NAT,
         "Quantum channels"),
        (0.25, 0.30, r"$\mathbf{Set}$", "#95A5A6",
         "Sets + functions"),
        (0.75, 0.30, r"$\mathbf{Meas}$", "#95A5A6",
         "Measurable spaces"),
        (0.5, 0.10, r"$\mathbf{Phys}$", C_MONOIDAL,
         "Physical theories"),
    ]

    for x, y, label, col, desc in hierarchy:
        cat_node(ax, x, y, label, col, 0.08)
        ax.text(x + 0.12, y, desc, ha="left", va="center",
                fontsize=7, color=C_TEXT, family="serif",
                style="italic", alpha=0.7)

    # Arrows (forgetful functors, inclusions)
    connections = [
        (0.5, 0.82, 0.5, 0.78, r"$\mathbf{F}$", C_FUNCTOR),
        (0.42, 0.65, 0.30, 0.56, "", C_ARROW),
        (0.58, 0.65, 0.70, 0.56, "", C_ARROW),
        (0.25, 0.42, 0.25, 0.38, "", C_ARROW),
        (0.75, 0.42, 0.75, 0.38, "", C_ARROW),
        (0.30, 0.24, 0.45, 0.15, "", C_ARROW),
        (0.70, 0.24, 0.55, 0.15, "", C_ARROW),
    ]
    for x0, y0, x1, y1, lbl, col in connections:
        morphism_arrow(ax, x0, y0, x1, y1, lbl, col, fontsize=10)

    ax.set_title("(d) Categorical Hierarchy",
                 fontsize=13, fontweight="bold", family="serif", pad=15)

    # ── Supertitle ──
    fig.suptitle(
        r"Functorial Structure: $\mathbf{F}: \mathbf{CogSys} \to \mathbf{Hilb_{CIIR}}$",
        fontsize=16, fontweight="bold", family="serif", y=1.0,
        color=C_TEXT
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("manuscripts/ciir_monograph/figures/fig08_functorial_structure.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig08_functorial_structure.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 8 saved: fig08_functorial_structure.{png,pdf}")


if __name__ == "__main__":
    main()

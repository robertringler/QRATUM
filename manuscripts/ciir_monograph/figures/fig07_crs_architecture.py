#!/usr/bin/env python3
"""
Figure 7: CRS 8-Layer Architecture
====================================
Shows the Computational Reality Substrate layers:
  Layer 0: Graph Substrate
  Layer 1: Rewrite Rules
  Layer 2: Dynamical Evolution
  Layer 3: Emergent Spacetime
  Layer 4: Branching Structure
  Layer 5: Conservation Laws
  Layer 6: Observer Coupling
  Layer 7: Full Integration

Publication-quality, 300 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ── Colours ──
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
LAYER_COLS = [
    "#E74C3C",  # Layer 0 — red
    "#E67E22",  # Layer 1 — orange
    "#F39C12",  # Layer 2 — yellow-orange
    "#27AE60",  # Layer 3 — green
    "#2980B9",  # Layer 4 — blue
    "#8E44AD",  # Layer 5 — purple
    "#1ABC9C",  # Layer 6 — teal
    "#2C3E50",  # Layer 7 — dark blue
]
C_ARROW    = "#7F8C8D"


def layer_box(ax, y, layer_num, name, description, module, colour, width=0.85):
    """Draw a layer box."""
    h = 0.08
    x = 0.5
    box = FancyBboxPatch(
        (x - width/2, y - h/2), width, h,
        boxstyle="round,pad=0.005",
        facecolor=colour, edgecolor=colour,
        alpha=0.10, linewidth=0, zorder=2
    )
    ax.add_patch(box)
    border = FancyBboxPatch(
        (x - width/2, y - h/2), width, h,
        boxstyle="round,pad=0.005",
        facecolor="none", edgecolor=colour,
        linewidth=2.0, zorder=3
    )
    ax.add_patch(border)

    # Layer number badge
    badge = plt.Circle((x - width/2 + 0.04, y), 0.025,
                        facecolor=colour, edgecolor="white",
                        linewidth=1.5, zorder=4)
    ax.add_patch(badge)
    ax.text(x - width/2 + 0.04, y, str(layer_num),
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", family="monospace", zorder=5)

    # Name
    ax.text(x - width/2 + 0.09, y + 0.005, name,
            ha="left", va="bottom", fontsize=12, fontweight="bold",
            color=colour, family="serif", zorder=4)

    # Description
    ax.text(x - width/2 + 0.09, y - 0.008, description,
            ha="left", va="top", fontsize=8, color=C_TEXT,
            family="serif", style="italic", zorder=4)

    # Module name
    ax.text(x + width/2 - 0.02, y, module,
            ha="right", va="center", fontsize=8, color=colour,
            family="monospace", alpha=0.7, zorder=4)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=300,
                             gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Left panel: Layer stack
    # ═══════════════════════════════════════════
    ax = axes[0]
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    ax.text(0.5, 0.98,
            "Computational Reality Substrate (CRS)",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color=C_TEXT, family="serif")
    ax.text(0.5, 0.94, "8-Layer Graph-Based Architecture",
            ha="center", va="top", fontsize=11, color=C_ARROW,
            family="serif", style="italic")

    layers = [
        (7, "Full Integration",
         "Compose all layers into unified CRS dynamics",
         "integration.py"),
        (6, "Observer Coupling",
         "Interface map Φ, CPTP channel, Born rule emergence",
         "observer.py"),
        (5, "Conservation Laws",
         "Noether currents, constraint preservation, unitarity",
         "conservation.py"),
        (4, "Branching Structure",
         "Quantum superposition, decoherence, branch selection",
         "branching.py"),
        (3, "Emergent Spacetime",
         "Metric from graph distance, causal structure, dimension",
         "spacetime.py"),
        (2, "Dynamical Evolution",
         "State update rules, Fokker–Planck, stochastic dynamics",
         "evolution.py"),
        (1, "Rewrite Rules",
         "Graph transformation, local updates, confluence",
         "rewrite.py"),
        (0, "Graph Substrate",
         "Nodes, edges, adjacency, weighted graph Σ = (V, E, w)",
         "graph.py"),
    ]

    y_positions = np.linspace(0.10, 0.85, 8)
    for i, (num, name, desc, mod) in enumerate(layers):
        layer_box(ax, y_positions[i], num, name, desc, mod,
                  LAYER_COLS[num])

    # Vertical arrows between layers
    for i in range(len(y_positions) - 1):
        ax.annotate(
            "", xy=(0.5, y_positions[i+1] - 0.04),
            xytext=(0.5, y_positions[i] + 0.04),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=1.5, mutation_scale=12, alpha=0.5)
        )

    # Side labels
    ax.text(0.02, 0.5, "EMERGENT\n↑\nFUNDAMENTAL",
            ha="center", va="center", fontsize=9, color=C_ARROW,
            family="serif", fontweight="bold", rotation=90, alpha=0.4)

    # ═══════════════════════════════════════════
    # Right panel: Graph visualization
    # ═══════════════════════════════════════════
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)

    # Create a sample CRS graph
    np.random.seed(123)
    N = 30  # nodes
    d = 2   # dimensions for layout

    # Position nodes
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + 0.3 * np.random.randn(N)
    px = radii * np.cos(angles) + 0.2 * np.random.randn(N)
    py = radii * np.sin(angles) + 0.2 * np.random.randn(N)

    # Node states (constraint values)
    C_vals = np.random.exponential(0.5, N)
    C_norm = C_vals / C_vals.max()

    # Draw edges (nearest neighbors)
    from scipy.spatial import Delaunay
    points = np.column_stack([px, py])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((simplex[i], simplex[j]))

        for i, j in edges:
            dist = np.sqrt((px[i]-px[j])**2 + (py[i]-py[j])**2)
            if dist < 1.5:
                # Edge weight ~ inverse distance
                w = max(0.1, 1.0 - dist / 2.0)
                ax2.plot([px[i], px[j]], [py[i], py[j]],
                         color=LAYER_COLS[0], lw=0.8 * w,
                         alpha=0.3 * w, zorder=1)
    except Exception:
        pass

    # Draw nodes colored by constraint value
    scatter = ax2.scatter(px, py, s=80, c=C_norm,
                          cmap="magma_r", vmin=0, vmax=1,
                          edgecolor="white", linewidth=0.8,
                          zorder=3)

    # Highlight low-constraint nodes (green ring)
    low_c = C_norm < 0.3
    ax2.scatter(px[low_c], py[low_c], s=120, facecolor="none",
                edgecolor=LAYER_COLS[3], linewidth=2.0, zorder=4)

    cb = plt.colorbar(scatter, ax=ax2, shrink=0.6,
                      label=r"$\mathcal{C}(\sigma)$ (normalised)")
    cb.ax.tick_params(labelsize=8)

    ax2.set_xlabel(r"$x^1$", fontsize=11, family="serif")
    ax2.set_ylabel(r"$x^2$", fontsize=11, family="serif")
    ax2.set_title("CRS Graph $\\Sigma = (V, E, w)$\n"
                  "Nodes coloured by constraint value",
                  fontsize=12, fontweight="bold", family="serif", pad=10)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.15)

    # Legend
    ax2.plot([], [], "o", color=LAYER_COLS[3], markersize=8,
             label=r"$\mathcal{C}(\sigma) < 0.3$ (near constraint surface)")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("manuscripts/ciir_monograph/figures/fig07_crs_architecture.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig07_crs_architecture.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 7 saved: fig07_crs_architecture.{png,pdf}")


if __name__ == "__main__":
    main()

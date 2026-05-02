#!/usr/bin/env python3
"""
Figure 9: Topological Manifold of Informational Filtering
===========================================================
GPAI STEM VISUALIZER — Prompt:
  "CIIR constraint bottleneck as fiber bundle projection from
   high-dimensional ontic manifold R through constraint manifold M
   to finite-dimensional epistemic Hilbert space H.
   Show information loss via dimensional collapse, fiber structure,
   and the CPTP channel as projection map. Scientific Diagram Style."

Design choices:
  - 3D surface pair showing high-dim R projected to low-dim H
  - Fiber bundle structure with vertical fibers (gauge orbits)
  - Gradient colouring by constraint curvature κ
  - Information funnel showing dimensional collapse
  - Nature/Science journal colour palette, 300 DPI

Publication-quality, 4K-equivalent at 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# ── Nature / Science colour palette ──────────────────────────────
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
C_ONTIC    = "#2C3E50"
C_CONSTR   = "#8E44AD"
C_IFACE    = "#E74C3C"
C_EPISTEMIC = "#2980B9"
C_FIBER    = "#F39C12"
C_INFO     = "#1ABC9C"
C_ARROW    = "#7F8C8D"
C_LOSS     = "#E74C3C"


def main():
    fig = plt.figure(figsize=(20, 14), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════════════════════════
    # Panel A: 3D fiber bundle — ontic manifold R with fibers
    # ═══════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_facecolor(C_BG)

    # High-dimensional ontic surface (represented as a curved 2-manifold
    # embedded in 3D, with fibers showing the "extra" dimensions)
    u = np.linspace(-2, 2, 80)
    v = np.linspace(-2, 2, 80)
    U, V = np.meshgrid(u, v)

    # Complex topological surface representing R
    W_ontic = 0.4 * np.sin(1.5 * U) * np.cos(1.5 * V) + 0.2 * np.cos(U + V)

    # Colour by "constraint curvature" κ(m)
    kappa = np.sqrt((np.gradient(W_ontic, axis=0))**2
                    + (np.gradient(W_ontic, axis=1))**2)
    kappa_norm = kappa / (kappa.max() + 1e-10)

    ax1.plot_surface(U, V, W_ontic, facecolors=cm.viridis(kappa_norm),
                     alpha=0.45, edgecolor="none", antialiased=True,
                     zorder=1)

    # Draw vertical fibers at selected points (gauge orbits / extra dimensions)
    fiber_points_u = np.linspace(-1.5, 1.5, 6)
    fiber_points_v = np.linspace(-1.5, 1.5, 6)
    for fu in fiber_points_u:
        for fv in fiber_points_v[::2]:
            u_idx = np.argmin(np.abs(u - fu))
            v_idx = np.argmin(np.abs(v - fv))
            z_base = W_ontic[v_idx, u_idx]
            fiber_z = np.linspace(z_base, z_base + 0.8, 15)
            ax1.plot([fu]*15, [fv]*15, fiber_z,
                     color=C_FIBER, lw=0.8, alpha=0.4, zorder=2)
            ax1.scatter([fu], [fv], [z_base + 0.8],
                        color=C_FIBER, s=8, alpha=0.5, zorder=3)

    # Projection arrows downward (representing the interface map Φ)
    for fu in [-1, 0, 1]:
        for fv in [-1, 0, 1]:
            u_idx = np.argmin(np.abs(u - fu))
            v_idx = np.argmin(np.abs(v - fv))
            z_top = W_ontic[v_idx, u_idx] + 0.4
            ax1.quiver(fu, fv, z_top, 0, 0, -0.6,
                       color=C_IFACE, arrow_length_ratio=0.25,
                       linewidth=1.5, alpha=0.6)

    ax1.set_xlabel(r"$r^1$", fontsize=10, family="serif", labelpad=3)
    ax1.set_ylabel(r"$r^2$", fontsize=10, family="serif", labelpad=3)
    ax1.set_zlabel(r"fiber dim", fontsize=9, family="serif", labelpad=3)
    ax1.set_title(
        r"(a) Fiber Bundle: $\pi: \mathcal{R} \to \mathcal{M}$"
        "\nVertical fibers = unobservable gauge orbits",
        fontsize=11, fontweight="bold", family="serif", pad=8)
    ax1.view_init(elev=22, azim=-55)

    # ═══════════════════════════════════════════════════════════════
    # Panel B: Information funnel — dimensional collapse
    # ═══════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(C_BG)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axis("off")

    # Draw information funnel
    # Top: wide (high-dimensional R)
    # Middle: constraint bottleneck (M)
    # Bottom: narrow (finite-dimensional H)

    n_lines = 40
    for i in range(n_lines):
        t = np.linspace(0, 1, 100)
        # Each "dimension" starts wide, narrows through bottleneck
        x_start = 0.5 + (i - n_lines/2) * 0.018
        x_end = 0.5 + (i - n_lines/2) * 0.004
        # Sigmoid-shaped convergence
        x_path = x_start + (x_end - x_start) / (1 + np.exp(-12 * (t - 0.5)))
        y_path = 1.0 - t

        # Color by whether this dimension survives
        surviving = abs(i - n_lines/2) < n_lines * 0.15
        if surviving:
            colour = C_EPISTEMIC
            alpha = 0.7
            lw = 1.5
        else:
            colour = C_LOSS
            alpha = 0.2
            lw = 0.6
        ax2.plot(x_path, y_path, color=colour, lw=lw, alpha=alpha)

    # Labels for strata
    ax2.text(0.5, 1.05, r"$\mathcal{R}$: Reality Space",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=C_ONTIC, family="serif")
    ax2.text(0.5, 1.0, r"$\dim = \infty$ (or very large)",
             ha="center", va="bottom", fontsize=8, color=C_ONTIC,
             family="serif", style="italic")

    # Constraint bottleneck zone
    from matplotlib.patches import Rectangle
    bottle = Rectangle((0.25, 0.42), 0.50, 0.16,
                        facecolor=C_CONSTR, alpha=0.08,
                        edgecolor=C_CONSTR, linewidth=2.0,
                        linestyle="--", zorder=0)
    ax2.add_patch(bottle)
    ax2.text(0.78, 0.50,
             r"$(\mathcal{M}, g, \hat{C})$"
             "\nConstraint\nBottleneck",
             ha="left", va="center", fontsize=10, fontweight="bold",
             color=C_CONSTR, family="serif")
    ax2.text(0.78, 0.42,
             r"$\kappa_{\max} < \infty$",
             ha="left", va="top", fontsize=9, color=C_CONSTR,
             family="serif", style="italic")

    # Bottom
    ax2.text(0.5, -0.05, r"$\mathcal{H}$: Cognitive State Space",
             ha="center", va="top", fontsize=12, fontweight="bold",
             color=C_EPISTEMIC, family="serif")
    ax2.text(0.5, -0.10,
             r"$\dim \leq \exp(\kappa_{\max} \cdot \mathrm{vol}(\mathcal{M}))$",
             ha="center", va="top", fontsize=9, color=C_EPISTEMIC,
             family="serif", style="italic")

    # Information loss annotation
    ax2.annotate(
        "Information\nlost", xy=(0.30, 0.30), xytext=(0.08, 0.20),
        fontsize=9, color=C_LOSS, family="serif", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_LOSS, lw=1.5))

    ax2.annotate(
        "Information\npreserved", xy=(0.50, 0.20), xytext=(0.08, 0.05),
        fontsize=9, color=C_EPISTEMIC, family="serif", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_EPISTEMIC, lw=1.5))

    # Arrow showing Φ
    ax2.annotate(
        "", xy=(0.5, -0.02), xytext=(0.5, 0.95),
        arrowprops=dict(arrowstyle="-|>", color=C_IFACE,
                        lw=3.0, mutation_scale=20, alpha=0.3),
        zorder=0)
    ax2.text(0.95, 0.70, r"$\Phi$",
             ha="center", va="center", fontsize=20, fontweight="bold",
             color=C_IFACE, family="serif", alpha=0.4)

    ax2.set_title(
        "(b) Informational Filtering: Dimensional Collapse",
        fontsize=11, fontweight="bold", family="serif", pad=10)

    # ═══════════════════════════════════════════════════════════════
    # Panel C: Constraint filtering spectrum
    # ═══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(C_BG)

    # Eigenvalue spectrum of the constraint operator Ĉ
    np.random.seed(42)
    n_modes = 50
    # Constraint eigenvalues (sorted, decaying)
    eigenvalues = np.sort(np.random.exponential(1.0, n_modes))[::-1]

    # Plot spectrum
    ax3.bar(range(n_modes), eigenvalues, color=C_CONSTR, alpha=0.6,
            edgecolor=C_CONSTR, linewidth=0.5)

    # Threshold line (information passes below this)
    threshold = eigenvalues[8]  # top ~8 modes pass through
    ax3.axhline(y=threshold, color=C_IFACE, lw=2.5, ls="--",
                label=r"Constraint threshold $\lambda_{\mathrm{cut}}$")

    # Shade passing modes
    for i in range(n_modes):
        if eigenvalues[i] >= threshold:
            ax3.bar(i, eigenvalues[i], color=C_EPISTEMIC, alpha=0.7,
                    edgecolor=C_EPISTEMIC, linewidth=0.5)

    ax3.set_xlabel("Mode index $k$", fontsize=11, family="serif")
    ax3.set_ylabel(r"Eigenvalue $\lambda_k(\hat{C})$", fontsize=11,
                   family="serif")
    ax3.set_title(
        r"(c) Constraint Spectrum: $\hat{C}$ Eigenvalues",
        fontsize=11, fontweight="bold", family="serif", pad=10)

    # Annotations
    ax3.annotate("Modes transmitted\nto $\\mathcal{H}$",
                 xy=(3, eigenvalues[3]), xytext=(15, eigenvalues[0]*0.95),
                 fontsize=9, color=C_EPISTEMIC, family="serif",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_EPISTEMIC,
                                 lw=1.5))
    ax3.annotate("Modes filtered out\n(information loss)",
                 xy=(25, eigenvalues[25]),
                 xytext=(30, eigenvalues[0]*0.5),
                 fontsize=9, color=C_LOSS, family="serif",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_LOSS,
                                 lw=1.5))

    ax3.text(40, eigenvalues[0]*0.8,
             r"$\dim(\mathcal{H}) = |\{k : \lambda_k \geq \lambda_{\mathrm{cut}}\}|$",
             fontsize=9, color=C_TEXT, family="serif",
             bbox=dict(facecolor="white", edgecolor=C_CONSTR,
                       boxstyle="round,pad=0.3", alpha=0.9))
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.15)

    # ═══════════════════════════════════════════════════════════════
    # Panel D: Topological manifold with projection map
    # ═══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.set_facecolor(C_BG)

    # Upper manifold: Reality space R (complex, wiggly)
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u, v)

    # Deformed sphere (representing ontic manifold)
    R_surf = 1.5 + 0.3 * np.sin(3*U) * np.sin(2*V) + 0.15 * np.cos(5*U + 3*V)
    X1 = R_surf * np.sin(V) * np.cos(U)
    Y1 = R_surf * np.sin(V) * np.sin(U)
    Z1 = R_surf * np.cos(V) + 2.0  # shift up

    # Constraint curvature on the surface
    kappa_surf = np.sqrt((np.gradient(R_surf, axis=0))**2
                         + (np.gradient(R_surf, axis=1))**2)
    kappa_n = kappa_surf / (kappa_surf.max() + 1e-10)

    ax4.plot_surface(X1, Y1, Z1, facecolors=cm.plasma(kappa_n),
                     alpha=0.35, edgecolor="none", antialiased=True)

    # Lower manifold: Interface image (smoother, smaller)
    R_low = 0.8 + 0.05 * np.sin(3*U) * np.sin(2*V)
    X2 = R_low * np.sin(V) * np.cos(U)
    Y2 = R_low * np.sin(V) * np.sin(U)
    Z2 = R_low * np.cos(V) - 1.5  # shift down

    ax4.plot_surface(X2, Y2, Z2, color=C_EPISTEMIC, alpha=0.25,
                     edgecolor=C_EPISTEMIC, linewidth=0.2, antialiased=True)

    # Projection lines from R to H
    n_proj = 12
    proj_u = np.linspace(0.3, 5.5, n_proj)
    proj_v = np.linspace(0.5, 2.5, n_proj//2)
    for pu in proj_u:
        for pv in proj_v:
            u_idx = int(pu / (2*np.pi) * 79) % 80
            v_idx = int(pv / np.pi * 39) % 40
            ax4.plot([X1[v_idx, u_idx], X2[v_idx, u_idx]],
                     [Y1[v_idx, u_idx], Y2[v_idx, u_idx]],
                     [Z1[v_idx, u_idx], Z2[v_idx, u_idx]],
                     color=C_IFACE, lw=0.5, alpha=0.25, ls="--")

    # Labels
    ax4.text(0, 0, 3.8, r"$\mathcal{R}$", fontsize=16, fontweight="bold",
             color=C_ONTIC, family="serif", ha="center")
    ax4.text(0, 0, -3.0, r"$\mathcal{H}$", fontsize=16, fontweight="bold",
             color=C_EPISTEMIC, family="serif", ha="center")
    ax4.text(1.5, 1.5, 0.5, r"$\Phi$", fontsize=14, fontweight="bold",
             color=C_IFACE, family="serif")

    ax4.set_title(
        r"(d) Projection: $\Phi: \mathcal{R} \twoheadrightarrow "
        r"\mathcal{B}(\mathcal{H}_\mathcal{M})$"
        "\nTopological complexity collapse",
        fontsize=11, fontweight="bold", family="serif", pad=8)
    ax4.view_init(elev=15, azim=-60)
    ax4.set_axis_off()

    # ═══════════════════════════════════════════════════════════════
    # Supertitle
    # ═══════════════════════════════════════════════════════════════
    fig.suptitle(
        "Topological Manifold of Informational Filtering\n"
        r"$\pi: (\mathcal{R}, d_\mathcal{R}) \;\to\; "
        r"(\mathcal{M}, g) \;\to\; "
        r"\mathfrak{D}(\mathcal{H})$",
        fontsize=16, fontweight="bold", family="serif", y=0.99,
        color=C_TEXT
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("manuscripts/ciir_monograph/figures/fig09_informational_filtering.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig09_informational_filtering.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 9 saved: fig09_informational_filtering.{png,pdf}")


if __name__ == "__main__":
    main()

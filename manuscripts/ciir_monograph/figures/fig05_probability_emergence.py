#!/usr/bin/env python3
"""
Figure 5: Emergence of Probability from Constraint Minimization
================================================================
Shows:
  (a) Boltzmann measure e^{-β C(σ)} concentrating on C⁻¹(0) as β → ∞
  (b) Maximum entropy derivation
  (c) Large deviation principle / rate function

Publication-quality, 300 DPI.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

# ── Colours ──
C_BG       = "#FAFAFA"
C_TEXT     = "#1A1A2E"
C_PROB     = "#8E44AD"
C_CONSTR   = "#E74C3C"
C_ENTROPY  = "#2980B9"
C_BETA     = "#27AE60"
C_LDP      = "#E67E22"


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig.patch.set_facecolor(C_BG)

    # ═══════════════════════════════════════════
    # Panel A: Constraint functional
    # ═══════════════════════════════════════════
    ax = axes[0, 0]
    ax.set_facecolor(C_BG)

    sigma = np.linspace(-3, 3, 500)
    C = (sigma**2 - 1)**2  # double well

    ax.plot(sigma, C, color=C_CONSTR, lw=2.5,
            label=r"$\mathcal{C}(\sigma) = (\sigma^2 - 1)^2$")
    ax.fill_between(sigma, 0, C, alpha=0.08, color=C_CONSTR)

    # Mark minima
    ax.plot([-1, 1], [0, 0], "o", color=C_BETA, markersize=10, zorder=5)
    ax.text(-1, -0.15, r"$\sigma^*_1$", ha="center", fontsize=11,
            color=C_BETA, family="serif")
    ax.text(1, -0.15, r"$\sigma^*_2$", ha="center", fontsize=11,
            color=C_BETA, family="serif")

    ax.set_xlabel(r"$\sigma$", fontsize=12, family="serif")
    ax.set_ylabel(r"$\mathcal{C}(\sigma)$", fontsize=12, family="serif")
    ax.set_title("(a) Constraint Functional",
                 fontsize=13, fontweight="bold", family="serif")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.3, 4)

    # ═══════════════════════════════════════════
    # Panel B: Boltzmann measure for different β
    # ═══════════════════════════════════════════
    ax = axes[0, 1]
    ax.set_facecolor(C_BG)

    betas = [0.5, 2, 5, 20, 100]
    colours_beta = plt.cm.viridis(np.linspace(0.2, 0.9, len(betas)))

    for beta, col in zip(betas, colours_beta):
        log_unnorm = -beta * C
        log_Z = logsumexp(log_unnorm)
        mu = np.exp(log_unnorm - log_Z)
        # Normalize for plot
        mu = mu / (np.sum(mu) * (sigma[1] - sigma[0]))
        ax.plot(sigma, mu, lw=2.0, color=col,
                label=rf"$\beta = {beta}$", alpha=0.85)

    ax.set_xlabel(r"$\sigma$", fontsize=12, family="serif")
    ax.set_ylabel(r"$\mu_\mathcal{C}(\sigma)$", fontsize=12, family="serif")
    ax.set_title(r"(b) Constraint Measure $\mu_\mathcal{C} \propto e^{-\beta \mathcal{C}}$",
                 fontsize=13, fontweight="bold", family="serif")
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)

    # Arrow showing β → ∞ concentration
    ax.annotate(r"$\beta \to \infty$: $\mu \to \delta_{\mathcal{C}^{-1}(0)}$",
                xy=(1, 3.5), xytext=(2, 3.5),
                fontsize=9, color=C_PROB, family="serif",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PROB, lw=1.5))

    # ═══════════════════════════════════════════
    # Panel C: Rate function (LDP)
    # ═══════════════════════════════════════════
    ax = axes[1, 0]
    ax.set_facecolor(C_BG)

    # Rate function I(σ) = C(σ) - min C
    I = C - np.min(C)

    ax.plot(sigma, I, color=C_LDP, lw=2.5,
            label=r"$I(\sigma) = \mathcal{C}(\sigma) - \inf \mathcal{C}$")
    ax.fill_between(sigma, 0, I, alpha=0.08, color=C_LDP)

    # Large deviation bound annotation
    ax.axhline(y=1.0, color=C_TEXT, lw=1, ls="--", alpha=0.3)
    ax.text(2.5, 1.1,
            r"$\Pr(\sigma \in A) \asymp e^{-\beta \inf_{\sigma \in A} I(\sigma)}$",
            fontsize=9, color=C_LDP, family="serif", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=C_LDP,
                      boxstyle="round,pad=0.3", alpha=0.9))

    # Shade a "rare event" region
    mask = (sigma > 1.5) & (sigma < 2.5)
    ax.fill_between(sigma[mask], 0, I[mask], alpha=0.25, color=C_LDP)
    ax.text(2.0, 0.3, r"$A$", fontsize=14, color=C_LDP,
            ha="center", family="serif", fontweight="bold")

    ax.set_xlabel(r"$\sigma$", fontsize=12, family="serif")
    ax.set_ylabel(r"Rate function $I(\sigma)$", fontsize=12, family="serif")
    ax.set_title("(c) Large Deviation Rate Function",
                 fontsize=13, fontweight="bold", family="serif")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.2, 4)

    # ═══════════════════════════════════════════
    # Panel D: Entropy vs constraint energy
    # ═══════════════════════════════════════════
    ax = axes[1, 1]
    ax.set_facecolor(C_BG)

    E_vals = np.linspace(0.01, 3, 100)
    # S(E) ∝ ln(volume of C ≤ E)
    # For double well: volume ∝ E^{1/2} for small E
    S_vals = np.log(1 + 2 * np.sqrt(E_vals))

    ax.plot(E_vals, S_vals, color=C_ENTROPY, lw=2.5,
            label=r"$S(E) = \ln \mathrm{vol}(\mathcal{C} \leq E)$")
    ax.fill_between(E_vals, 0, S_vals, alpha=0.08, color=C_ENTROPY)

    # Temperature = dE/dS
    dSdE = np.gradient(S_vals, E_vals)
    beta_eff = dSdE
    ax2 = ax.twinx()
    ax2.plot(E_vals, 1/beta_eff, color=C_PROB, lw=1.5, ls="--",
             label=r"$T_\mathrm{eff} = (dS/dE)^{-1}$", alpha=0.7)
    ax2.set_ylabel(r"Effective temperature $T_\mathrm{eff}$",
                   fontsize=11, family="serif", color=C_PROB)
    ax2.tick_params(axis="y", colors=C_PROB)

    ax.set_xlabel(r"Constraint energy $E = \langle \mathcal{C} \rangle$",
                  fontsize=12, family="serif")
    ax.set_ylabel(r"Entropy $S(E)$", fontsize=12, family="serif",
                  color=C_ENTROPY)
    ax.tick_params(axis="y", colors=C_ENTROPY)
    ax.set_title("(d) Microcanonical Entropy",
                 fontsize=13, fontweight="bold", family="serif")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.2)

    # ── Supertitle ──
    fig.suptitle(
        "Emergence of Probability from Constraint Minimization",
        fontsize=16, fontweight="bold", family="serif", y=1.0,
        color=C_TEXT
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("manuscripts/ciir_monograph/figures/fig05_probability_emergence.png",
                dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.savefig("manuscripts/ciir_monograph/figures/fig05_probability_emergence.pdf",
                bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("✓ Figure 5 saved: fig05_probability_emergence.{png,pdf}")


if __name__ == "__main__":
    main()

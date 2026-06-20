"""RQ-3 — Fraction of realistic MIMO channels with kappa(G) > 77 (the POC
resolvent convergence limit), and the resulting digital-cleanup fallback rate.

Question (POC-APPS-1.0, RQ-3): the POC's resolvent loop converges in <= 40
transits only if (alpha ||G||)^40 is small, i.e. kappa(G) below a threshold.
With alpha = 0.9/||G||, the Neumann/Richardson contraction factor per transit
is approximately 1 - 1/kappa(G), so reaching tolerance eps in 40 transits needs

    (1 - 1/kappa)^40 <= eps   =>   kappa <= -40 / ln(eps)    (approx).

For eps = 1e-2 this gives the kappa ~ 77 figure quoted in POC-APPS-1.0. We
measure, over realistic massive-MIMO channel ensembles, what fraction of
channel realizations exceed this — i.e. the rate at which the POC must fall
back to the hybrid digital-cleanup loop.

Channel models:
  - i.i.d. Rayleigh (rich scattering, best case for conditioning)
  - spatially correlated Rayleigh (Kronecker model, exponential correlation
    rho_corr) — realistic for ULA base stations; higher correlation -> worse
    conditioning
  - line-of-sight contamination (Rician K-factor) — a few dominant paths,
    worsening conditioning

G = H H^H + lambda I  (regularized Gram, n_UE x n_UE), with regularization
lambda = noise floor. We sweep n_UE/n_BS loading and correlation.

Outputs:
  results/rq3_mimo_conditioning.json
  figures/rq3_kappa_cdf.png
"""
from __future__ import annotations

import json
import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEED = 23
N_BS = 64
N_TRIALS = 2000
TOL = 1e-2
KAPPA_LIMIT = -40.0 / np.log(TOL)  # ~ 8.69 ... wait: see note in analysis

# Loading factors n_UE / n_BS and correlation coefficients to sweep.
UE_GRID = [8, 16, 32, 48]          # n_UE
CORR_GRID = [0.0, 0.3, 0.5, 0.7]   # exponential spatial correlation
RICIAN_K_DB = [None, 0, 6]          # None=Rayleigh; else Rician K-factor (dB)


def exp_corr_matrix(n, rho):
    """Exponential correlation model R[i,j] = rho^|i-j| (Hermitian)."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sqrtm_herm(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    return V @ np.diag(np.sqrt(w)) @ V.conj().T


# Cache the correlation-matrix square root per (n_bs, corr) — it is identical
# across all trials, so computing it once instead of per-trial removes the
# dominant cost (a 64x64 eigendecomposition repeated thousands of times).
_SQRT_CACHE: dict = {}


def _corr_sqrt(n_bs, corr):
    key = (n_bs, corr)
    if key not in _SQRT_CACHE:
        _SQRT_CACHE[key] = sqrtm_herm(exp_corr_matrix(n_bs, corr))
    return _SQRT_CACHE[key]


def gen_channel(n_ue, n_bs, corr, rician_k_db, rng):
    """n_UE x n_BS channel with optional Tx correlation and Rician LOS."""
    Hiid = (rng.standard_normal((n_ue, n_bs)) +
            1j * rng.standard_normal((n_ue, n_bs))) / np.sqrt(2)
    if corr > 0:
        Hiid = Hiid @ _corr_sqrt(n_bs, corr)
    if rician_k_db is not None:
        K = 10 ** (rician_k_db / 10.0)
        # deterministic LOS component (rank-1-ish steering)
        angles = rng.uniform(-np.pi / 3, np.pi / 3, size=n_ue)
        Hlos = np.exp(1j * np.pi * np.outer(np.sin(angles), np.arange(n_bs)))
        H = np.sqrt(K / (K + 1)) * Hlos + np.sqrt(1.0 / (K + 1)) * Hiid
    else:
        H = Hiid
    return H


def transits_richardson(kappa, tol=TOL):
    """Simple Richardson/Neumann resolvent iterations to reach tol.
    contraction per transit c = 1 - 1/kappa  =>  c^t <= tol. This is the POC's
    basic resolvent mode (loop with fixed gain)."""
    c = 1.0 - 1.0 / kappa
    if c <= 0:
        return 1
    return int(np.ceil(np.log(tol) / np.log(c)))


def transits_chebyshev(kappa, tol=TOL):
    """Chebyshev-accelerated iterations (the POC's cross-coupled-loop Chebyshev
    mode, Vol I Sec 6.2). For an SPD system the error after t steps is bounded
    by 2 * r^t with r = (sqrt(kappa)-1)/(sqrt(kappa)+1), giving the classical
    sqrt(kappa) iteration scaling — a quadratic improvement over Richardson."""
    sk = np.sqrt(kappa)
    r = (sk - 1.0) / (sk + 1.0)
    if r <= 0:
        return 1
    # 2 r^t <= tol  =>  t >= ln(tol/2)/ln(r)
    return int(np.ceil(np.log(tol / 2.0) / np.log(r)))


def analyze(n_ue, corr, rician_k_db):
    rng = np.random.default_rng(
        SEED + n_ue * 13 + int(corr * 100) +
        (0 if rician_k_db is None else (rician_k_db + 1) * 1000)
    )
    snr_db = 15.0
    lam = 1.0 / (10 ** (snr_db / 10.0)) * n_ue  # regularization ~ noise power
    kappas = []
    t_rich = []
    t_cheb = []
    for _ in range(N_TRIALS):
        H = gen_channel(n_ue, N_BS, corr, rician_k_db, rng)
        G = H @ H.conj().T + lam * np.eye(n_ue)
        ev = np.linalg.eigvalsh(G)
        kappa = float(ev[-1] / ev[0])
        kappas.append(kappa)
        t_rich.append(transits_richardson(kappa))
        t_cheb.append(transits_chebyshev(kappa))
    kappas = np.array(kappas)
    t_rich = np.array(t_rich)
    t_cheb = np.array(t_cheb)
    return {
        "n_ue": n_ue, "corr": corr, "rician_k_db": rician_k_db,
        "kappa_median": float(np.median(kappas)),
        "kappa_p90": float(np.percentile(kappas, 90)),
        "kappa_p99": float(np.percentile(kappas, 99)),
        # Fallback = cannot converge within the 40-transit loop-depth limit.
        "fallback_richardson": float(np.mean(t_rich > 40)),
        "fallback_chebyshev": float(np.mean(t_cheb > 40)),
        "mean_transits_richardson": float(np.mean(np.minimum(t_rich, 40))),
        "mean_transits_chebyshev": float(np.mean(np.minimum(t_cheb, 40))),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    grid = {}
    kap_rich = 1.0 / (1.0 - TOL ** (1.0 / 40))
    kap_cheb = ((1 + (TOL/2) ** (1/40)) / (1 - (TOL/2) ** (1/40))) ** 2
    print(f"n_BS={N_BS}, trials={N_TRIALS}, tol={TOL}")
    print(f"40-transit kappa limit: Richardson ~ {kap_rich:.1f}, "
          f"Chebyshev ~ {kap_cheb:.0f}")
    print(f"\n{'n_UE':>5} {'corr':>5} {'Rician':>7} {'kap_med':>9} "
          f"{'kap_p90':>9} {'fb_Rich':>8} {'fb_Cheb':>8} {'t_Cheb':>7}")
    print("-" * 70)
    for n_ue in UE_GRID:
        for corr in CORR_GRID:
            for rk in RICIAN_K_DB:
                r = analyze(n_ue, corr, rk)
                key = f"ue{n_ue}_corr{corr}_rk{rk}"
                grid[key] = r
                rk_s = "Rayl" if rk is None else f"{rk}dB"
                print(f"{n_ue:>5} {corr:>5} {rk_s:>7} "
                      f"{r['kappa_median']:>9.1f} {r['kappa_p90']:>9.1f} "
                      f"{r['fallback_richardson']*100:>7.1f}% "
                      f"{r['fallback_chebyshev']*100:>7.1f}% "
                      f"{r['mean_transits_chebyshev']:>7.1f}")

    out = os.path.join(RESULTS_DIR, "rq3_mimo_conditioning.json")
    with open(out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nWrote {out}")
    _plot(grid)


def _plot(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: Richardson vs Chebyshev fallback rate vs loading (Rayleigh, corr=0.3).
    ax = axes[0]
    corr = 0.3
    rich = [grid[f"ue{ue}_corr{corr}_rkNone"]["fallback_richardson"] * 100
            for ue in UE_GRID]
    cheb = [grid[f"ue{ue}_corr{corr}_rkNone"]["fallback_chebyshev"] * 100
            for ue in UE_GRID]
    ax.plot([ue / N_BS for ue in UE_GRID], rich, "r-o",
            label="Richardson resolvent")
    ax.plot([ue / N_BS for ue in UE_GRID], cheb, "g-s",
            label="Chebyshev-accelerated")
    ax.set_xlabel("loading n_UE / n_BS")
    ax.set_ylabel("digital-cleanup fallback rate (%)")
    ax.set_title("RQ-3: POC fallback rate, Richardson vs Chebyshev\n"
                 f"(Rayleigh, corr={corr}, n_BS=64, 40-transit limit)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: condition-number percentiles at full loading (n_UE=32).
    ax = axes[1]
    x = np.arange(len(CORR_GRID))
    w = 0.25
    for off, rk in zip([-w, 0, w], RICIAN_K_DB):
        rk_s = "Rayleigh" if rk is None else f"Rician {rk}dB"
        ys = [grid[f"ue32_corr{corr}_rk{rk}"]["kappa_p90"] for corr in CORR_GRID]
        ax.bar(x + off, ys, w, label=rk_s)
    ax.axhline(77, color="r", ls="--", label="kappa=77 (40-transit limit)")
    ax.set_xticks(x)
    ax.set_xticklabels(CORR_GRID)
    ax.set_xlabel("spatial correlation")
    ax.set_ylabel("kappa(G), 90th percentile")
    ax.set_title("RQ-3: 90th-pct condition number at full loading (n_UE=32)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "rq3_kappa_cdf.png")
    fig.savefig(p, dpi=130)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()

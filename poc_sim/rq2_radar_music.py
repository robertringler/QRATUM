"""RQ-2 — Does POC readout precision degrade radar DOA, or does the classical
angular-resolution limit bind first?

Question (POC-APPS-1.0, RQ-2): does ringdown spectroscopy achieve eigenvalue
resolution sufficient for MUSIC at n=64? An initial proxy check (gap vs
spectral resolution) showed the eigengap stays far above the 7-bit resolution
limit even at 0.5 deg separation — suggesting the POC spectral resolution is
NOT the bottleneck. We confirm that properly here by running MUSIC end to end
with a POC-precision-limited eigendecomposition and measuring ANGULAR RMSE
against fp64 MUSIC. The decisive comparison:

  - classical fp64 MUSIC: subject only to the angular (beamwidth ~ 2/n rad
    ~ 1.8 deg at n=64) and finite-snapshot limits.
  - POC MUSIC: additionally, the noise-subspace eigenvectors are read out at
    P bits (the POC measures the eigendecomposition with finite precision).

If POC RMSE tracks fp64 RMSE, the POC precision is not the binding constraint
(good for the POC). Where it diverges, we have found the precision-limited
regime. We sweep SNR and angular separation and also sweep P in {5,6,7,8}.

Outputs:
  results/rq2_radar_music.json
  figures/rq2_music_rmse.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_precision_model import apply_poc_readout  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEED = 17
N = 64
K = 2
N_SNAPSHOTS = 200
N_TRIALS = 100
ANGLE_GRID = np.linspace(-30, 40, 1401)  # 0.05 deg resolution pseudospectrum

SNR_GRID_DB = [0, 5, 10, 15, 20]
SEP_GRID_DEG = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
P_GRID = [5, 6, 7, 8]
BASE_ANGLE = 10.0


def steering(theta_deg, n):
    theta = np.deg2rad(theta_deg)
    return np.exp(1j * np.pi * np.arange(n) * np.sin(theta))


def sample_covariance(thetas, snr_db, n, n_snap, rng):
    sigma_n2 = 1.0 / (10 ** (snr_db / 10.0))
    A = np.column_stack([steering(t, n) for t in thetas])
    S = (rng.standard_normal((len(thetas), n_snap)) +
         1j * rng.standard_normal((len(thetas), n_snap))) / np.sqrt(2)
    Noise = (rng.standard_normal((n, n_snap)) +
             1j * rng.standard_normal((n, n_snap))) / np.sqrt(2) * np.sqrt(sigma_n2)
    X = A @ S + Noise
    return (X @ X.conj().T) / n_snap


# Precompute the steering matrix over the whole pseudospectrum angle grid once
# (shape n x n_angles). Reused across every trial: this turns the per-angle
# Python loop into a single vectorized matrix product.
_A_GRID = np.exp(
    1j * np.pi * np.outer(np.arange(N), np.sin(np.deg2rad(ANGLE_GRID)))
)


def music_spectrum(R, k, n, precision_bits, rng):
    """MUSIC pseudospectrum (vectorized over the angle grid). If
    precision_bits is not None, the eigenvectors are read out at P bits."""
    evals, evecs = np.linalg.eigh(R)  # ascending
    Un = evecs[:, : n - k]
    if precision_bits is not None:
        # POC reads the eigenvectors at finite precision (complex homodyne).
        Un = apply_poc_readout(Un, precision_bits, rng, dynamic_range=1.0)
    # denom(theta) = || Un^H a(theta) ||^2, computed for all angles at once.
    M = Un.conj().T @ _A_GRID            # (n-k) x n_angles
    denom = np.sum(np.abs(M) ** 2, axis=0)
    return 1.0 / np.maximum(denom, 1e-12)


def estimate_angles(spec, k):
    """Top-k peaks of the pseudospectrum (vectorized local-maxima search)."""
    interior = np.arange(1, len(spec) - 1)
    is_peak = (spec[interior] > spec[interior - 1]) & \
              (spec[interior] > spec[interior + 1])
    peak_idx = interior[is_peak]
    if len(peak_idx) == 0:
        return []
    top = peak_idx[np.argsort(spec[peak_idx])[::-1][:k]]
    return sorted(ANGLE_GRID[top].tolist())


def rmse_angles(est, true):
    if len(est) < len(true):
        # failed to find enough peaks -> max error
        return 90.0
    est = np.array(sorted(est))
    true = np.array(sorted(true))
    return float(np.sqrt(np.mean((est - true) ** 2)))


def analyze(snr_db, sep_deg, precision_bits):
    rng = np.random.default_rng(
        SEED + int(snr_db) * 131 + int(sep_deg * 10) +
        (0 if precision_bits is None else precision_bits * 7)
    )
    thetas = [BASE_ANGLE, BASE_ANGLE + sep_deg]
    errs = []
    for _ in range(N_TRIALS):
        R = sample_covariance(thetas, snr_db, N, N_SNAPSHOTS, rng)
        spec = music_spectrum(R, K, N, precision_bits, rng)
        est = estimate_angles(spec, K)
        errs.append(rmse_angles(est, thetas))
    errs = np.array(errs)
    # "resolved" if RMSE < sep/2 (the two peaks are distinct and near-correct)
    resolved = float(np.mean(errs < sep_deg / 2.0))
    return {"rmse_median": float(np.median(errs)),
            "rmse_mean": float(np.mean(errs)),
            "resolved_fraction": resolved}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    grid = {}
    print(f"n={N}, K={K}, snapshots={N_SNAPSHOTS}, trials={N_TRIALS}")
    print(f"beamwidth ~ {np.rad2deg(2.0/N):.2f} deg (classical resolution floor)\n")

    # Focus sweep: fix SNR=15 dB, vary separation, compare fp64 vs P bits.
    print("=== SNR=15 dB: RMSE (deg) vs separation, fp64 vs POC precision ===")
    header = f"{'sep°':>6} {'fp64':>8}" + "".join(f"{'P='+str(p):>8}" for p in P_GRID)
    print(header)
    for sep in SEP_GRID_DEG:
        ref = analyze(15, sep, None)
        row = f"{sep:>6} {ref['rmse_median']:>8.3f}"
        grid[f"snr15_sep{sep}_fp64"] = ref
        for P in P_GRID:
            r = analyze(15, sep, P)
            grid[f"snr15_sep{sep}_P{P}"] = r
            row += f"{r['rmse_median']:>8.3f}"
        print(row)

    # Full SNR x separation at fixed P=6 (the POC operating point).
    print("\n=== P=6 bits: resolved-fraction heatmap (SNR x separation) ===")
    print(f"{'SNR':>5}" + "".join(f"{s:>7}" for s in SEP_GRID_DEG))
    heat = np.zeros((len(SNR_GRID_DB), len(SEP_GRID_DEG)))
    heat_ref = np.zeros_like(heat)
    for i, snr in enumerate(SNR_GRID_DB):
        row = f"{snr:>5}"
        for j, sep in enumerate(SEP_GRID_DEG):
            r = analyze(snr, sep, 6)
            ref = analyze(snr, sep, None)
            grid[f"snr{snr}_sep{sep}_P6"] = r
            grid[f"snr{snr}_sep{sep}_fp64ref"] = ref
            heat[i, j] = r["resolved_fraction"]
            heat_ref[i, j] = ref["resolved_fraction"]
            row += f"{r['resolved_fraction']*100:>6.0f}%"
        print(row)

    out = os.path.join(RESULTS_DIR, "rq2_radar_music.json")
    with open(out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nWrote {out}")
    _plot(grid, heat, heat_ref)


def _plot(grid, heat, heat_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ref = [grid[f"snr15_sep{s}_fp64"]["rmse_median"] for s in SEP_GRID_DEG]
    ax.semilogy(SEP_GRID_DEG, ref, "k-x", lw=2, label="fp64 MUSIC")
    for P in P_GRID:
        ys = [grid[f"snr15_sep{s}_P{P}"]["rmse_median"] for s in SEP_GRID_DEG]
        ax.semilogy(SEP_GRID_DEG, ys, marker="o", label=f"POC P={P}")
    ax.axvline(np.rad2deg(2.0 / N), color="r", ls=":",
               label=f"beamwidth {np.rad2deg(2.0/N):.1f}°")
    ax.set_xlabel("angular separation (deg)")
    ax.set_ylabel("angle RMSE (deg, median)")
    ax.set_title("RQ-2: MUSIC angle RMSE, fp64 vs POC precision (SNR=15 dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    im = ax.imshow(heat - heat_ref, origin="lower", aspect="auto",
                   cmap="RdBu", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(SEP_GRID_DEG)))
    ax.set_xticklabels(SEP_GRID_DEG)
    ax.set_yticks(range(len(SNR_GRID_DB)))
    ax.set_yticklabels(SNR_GRID_DB)
    ax.set_xlabel("angular separation (deg)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("POC(P=6) minus fp64 resolved-fraction\n"
                 "(blue = POC worse; white = POC ties fp64)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{(heat[i,j]-heat_ref[i,j])*100:+.0f}",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="delta resolved fraction")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "rq2_music_rmse.png")
    fig.savefig(p, dpi=130)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()

"""RQ-1b — Linear memory capacity of the POC reservoir vs precision and n.

Motivation (a correction discovered during RQ-1): a LINEAR reservoir — the
POC's native f=identity mode — cannot solve NARMA-10 even at fp64 (NMSE ~0.3),
because NARMA-10 is nonlinear (product terms). The 0.01-0.02 literature figures
are for tanh reservoirs. So NARMA-10 is the WRONG benchmark for the POC's linear
loop. The RIGHT benchmark is the task linear reservoirs are provably good at:

  Short-Term Memory Capacity (Jaeger 2002):
    MC = sum_{k>=1} corr^2( y_k(t), u(t-k) )
  where y_k is the best linear reconstruction of the input delayed by k steps.
  For a linear reservoir of size n driven by i.i.d. input, MC <= n (tight bound).

This experiment measures how much of that theoretical capacity the POC retains
at P bits of readout precision — the genuinely POC-appropriate reservoir metric.

Outputs:
  results/rq1b_memory_capacity.json
  figures/rq1b_mc_vs_precision.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poc_precision_model import apply_poc_readout, shots_for_bits  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEED = 7
T_TOTAL = 6000
T_WASHOUT = 200
T_TRAIN = 3000
RIDGE = 1e-8
MAX_DELAY_FACTOR = 2  # probe delays up to 2n

N_GRID = [16, 32, 64, 128]
RHO_GRID = [0.7, 0.9, 0.95]
PREC_GRID = [4, 5, 6, 7, 8, 10]


def make_reservoir(n: int, rho: float, rng: np.random.Generator):
    W = rng.standard_normal((n, n))
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (rho / eig)
    W_in = rng.uniform(-0.5, 0.5, size=(n, 1))
    return W, W_in


def run_reservoir(W, W_in, u, alpha, precision_bits, rng):
    n = W.shape[0]
    x = np.zeros(n)
    states = np.zeros((len(u), n))
    for t in range(len(u)):
        pre = W @ x + W_in[:, 0] * u[t]
        if precision_bits is not None:
            pre = apply_poc_readout(pre, precision_bits, rng, dynamic_range=None)
        x = (1.0 - alpha) * x + alpha * pre
        states[t] = x
    return states


def memory_capacity(states, u, max_delay):
    """MC = sum_k corr^2(best linear reconstruction of u(t-k), u(t-k))."""
    tr = slice(T_WASHOUT, T_TRAIN)
    te = slice(T_TRAIN, len(u))
    X_tr = np.hstack([states[tr], np.ones((T_TRAIN - T_WASHOUT, 1))])
    X_te = np.hstack([states[te], np.ones((len(u) - T_TRAIN, 1))])
    A = X_tr.T @ X_tr + RIDGE * np.eye(X_tr.shape[1])

    mc = 0.0
    per_k = []
    for k in range(1, max_delay + 1):
        # target: input delayed by k
        tgt_tr = u[T_WASHOUT - 0:T_TRAIN]  # placeholder, shifted below
        tgt_tr = np.array([u[t - k] for t in range(T_WASHOUT, T_TRAIN)])
        tgt_te = np.array([u[t - k] for t in range(T_TRAIN, len(u))])
        w = np.linalg.solve(A, X_tr.T @ tgt_tr)
        pred = X_te @ w
        c = np.corrcoef(pred, tgt_te)[0, 1]
        c2 = float(c ** 2) if np.isfinite(c) else 0.0
        per_k.append(c2)
        mc += c2
    return mc, per_k


def evaluate(n, rho, precision_bits):
    rng = np.random.default_rng(SEED + n * 100 + int(rho * 100))
    u = rng.uniform(-0.5, 0.5, size=T_TOTAL)  # i.i.d. input for MC
    alpha = 1.0  # no leak: maximal memory
    W, W_in = make_reservoir(n, rho, rng)
    states = run_reservoir(W, W_in, u, alpha, precision_bits, rng)
    max_delay = MAX_DELAY_FACTOR * n
    return memory_capacity(states, u, max_delay)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    grid = {}
    print(f"{'n':>4} {'rho':>5} {'P':>5} {'MC':>9} {'MC/n':>7} {'shots':>10}")
    print("-" * 46)
    for n in N_GRID:
        for rho in RHO_GRID:
            mc_ref, _ = evaluate(n, rho, None)
            grid[f"n{n}_rho{rho}_fp64"] = {
                "n": n, "rho": rho, "precision": "fp64",
                "mc": mc_ref, "mc_over_n": mc_ref / n, "shots": 0,
            }
            print(f"{n:>4} {rho:>5} {'fp64':>5} {mc_ref:>9.3f} "
                  f"{mc_ref/n:>7.3f} {0:>10}")
            for P in PREC_GRID:
                mc, per_k = evaluate(n, rho, P)
                grid[f"n{n}_rho{rho}_P{P}"] = {
                    "n": n, "rho": rho, "precision": P,
                    "mc": mc, "mc_over_n": mc / n,
                    "shots": shots_for_bits(P),
                }
                print(f"{n:>4} {rho:>5} {P:>5} {mc:>9.3f} "
                      f"{mc/n:>7.3f} {shots_for_bits(P):>10}")

    out = os.path.join(RESULTS_DIR, "rq1b_memory_capacity.json")
    with open(out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nWrote {out}")
    _plot(grid)


def _plot(grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in N_GRID:
        ys = [grid[f"n{n}_rho0.9_P{P}"]["mc_over_n"] for P in PREC_GRID]
        ax.plot(PREC_GRID, ys, marker="o", label=f"n={n}")
        ref = grid[f"n{n}_rho0.9_fp64"]["mc_over_n"]
        ax.axhline(ref, ls="--", lw=0.8, alpha=0.5)
    ax.axvspan(5, 7, alpha=0.08, color="green")
    ax.set_xlabel("POC precision (bits)")
    ax.set_ylabel("memory capacity fraction MC / n")
    ax.set_title("RQ-1b: Linear reservoir memory capacity vs POC precision\n"
                 "(rho=0.9; dashed = fp64 ceiling; green = POC envelope)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "rq1b_mc_vs_precision.png")
    fig.savefig(p, dpi=130)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()

"""RQ-1 — Precision loss in linear reservoirs vs spectral radius rho and size n.

Question (POC-APPS-1.0, RQ-1): the reservoir-computing literature uses fp64.
The POC precision model predicts graceful degradation for rho < 0.9, but this
has not been validated. We validate it in simulation here, before any M0
hardware exists.

Method:
  - NARMA-10 benchmark task (standard echo-state-network benchmark).
  - Linear reservoir: x(t+1) = (1-a) x(t) + a (W_res x(t) + W_in u(t)).
    f = identity (the POC's native mode).
  - Sweep reservoir size n in {16, 32, 64, 128} and spectral radius
    rho(W_res) in {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}.
  - For each (n, rho), run an fp64 reference reservoir AND a POC reservoir in
    which BOTH the reservoir state readout and the W_res application are
    subject to the P-bit POC readout model, for P in {4, 5, 6, 7, 8, 10}.
  - Metric: NMSE of the trained linear readout on a held-out test set.

The readout W_out is trained by ridge regression on collected states. The
contractivity (echo-state property) requires rho < 1; we deliberately push to
rho = 0.99 to find where the POC precision model breaks (the contraction
forgets noise at rate rho per step, so steady-state noise ~ delta/(1-rho) —
Vol I Sec 10.3 robustness semantics — predicting blow-up as rho -> 1).

Outputs:
  results/rq1_reservoir_precision.json   (full grid)
  figures/rq1_nmse_vs_precision.png
  figures/rq1_nmse_vs_rho.png
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

SEED = 42
T_TOTAL = 6000
T_WASHOUT = 200
T_TRAIN = 3000
RIDGE = 1e-6

N_GRID = [16, 32, 64, 128]
RHO_GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
PREC_GRID = [4, 5, 6, 7, 8, 10]  # fp64 reference handled separately


def narma10(u: np.ndarray) -> np.ndarray:
    """Generate NARMA-10 target series from input u in [0, 0.5]."""
    y = np.zeros_like(u)
    for t in range(10, len(u)):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    return y


def make_reservoir(n: int, rho: float, rng: np.random.Generator):
    """Random reservoir scaled to target spectral radius rho; random input map."""
    W = rng.standard_normal((n, n))
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (rho / eig)
    W_in = rng.uniform(-0.5, 0.5, size=(n, 1))
    return W, W_in


def run_reservoir(
    W: np.ndarray,
    W_in: np.ndarray,
    u: np.ndarray,
    alpha: float,
    precision_bits: float | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Drive the linear reservoir; collect states. None precision => fp64."""
    n = W.shape[0]
    x = np.zeros(n)
    states = np.zeros((len(u), n))
    for t in range(len(u)):
        pre = W @ x + W_in[:, 0] * u[t]
        if precision_bits is not None:
            # POC applies W physically: the matvec result is read at P bits.
            pre = apply_poc_readout(pre, precision_bits, rng, dynamic_range=None)
        x = (1.0 - alpha) * x + alpha * pre
        states[t] = x
    return states


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2) / np.var(target))


def train_readout(states: np.ndarray, target: np.ndarray):
    """Ridge regression readout on [states, bias]."""
    X = np.hstack([states, np.ones((len(states), 1))])
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ target)
    return w


def apply_readout(states: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.hstack([states, np.ones((len(states), 1))])
    return X @ w


def evaluate(n: int, rho: float, precision_bits: float | None) -> float:
    rng = np.random.default_rng(SEED + n * 100 + int(rho * 100))
    u = rng.uniform(0.0, 0.5, size=T_TOTAL)
    y = narma10(u)
    alpha = 0.3
    W, W_in = make_reservoir(n, rho, rng)

    states = run_reservoir(W, W_in, u, alpha, precision_bits, rng)

    tr = slice(T_WASHOUT, T_TRAIN)
    te = slice(T_TRAIN, T_TOTAL)
    w = train_readout(states[tr], y[tr])
    pred_te = apply_readout(states[te], w)
    return nmse(pred_te, y[te])


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    grid = {}
    print(f"{'n':>4} {'rho':>5} {'P':>5} {'NMSE':>10} {'shots':>12}")
    print("-" * 42)
    for n in N_GRID:
        for rho in RHO_GRID:
            ref = evaluate(n, rho, None)
            grid[f"n{n}_rho{rho}_fp64"] = {
                "n": n, "rho": rho, "precision": "fp64",
                "nmse": ref, "shots": 0,
            }
            print(f"{n:>4} {rho:>5} {'fp64':>5} {ref:>10.5f} {0:>12}")
            for P in PREC_GRID:
                val = evaluate(n, rho, P)
                grid[f"n{n}_rho{rho}_P{P}"] = {
                    "n": n, "rho": rho, "precision": P,
                    "nmse": val, "shots": shots_for_bits(P),
                }
                print(f"{n:>4} {rho:>5} {P:>5} {val:>10.5f} "
                      f"{shots_for_bits(P):>12}")

    out = os.path.join(RESULTS_DIR, "rq1_reservoir_precision.json")
    with open(out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nWrote {out}")

    _plot(grid)


def _plot(grid: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: NMSE vs precision, one line per rho, at n=64.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for rho in RHO_GRID:
        ys = []
        for P in PREC_GRID:
            ys.append(grid[f"n64_rho{rho}_P{P}"]["nmse"])
        ax.semilogy(PREC_GRID, ys, marker="o", label=f"rho={rho}")
    ref64 = grid["n64_rho0.9_fp64"]["nmse"]
    ax.axhline(ref64, color="k", ls="--", lw=1, label="fp64 (rho=0.9)")
    ax.axhline(0.03, color="r", ls=":", lw=1.5, label="BM-R1 target 0.03")
    ax.axvspan(5, 7, alpha=0.08, color="green")
    ax.set_xlabel("POC precision (bits)")
    ax.set_ylabel("NMSE (test)")
    ax.set_title("RQ-1: Reservoir NMSE vs POC precision (n=64, NARMA-10)\n"
                 "green band = POC certified-precision envelope (5-7 bits)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "rq1_nmse_vs_precision.png")
    fig.savefig(p1, dpi=130)
    print(f"Wrote {p1}")

    # Fig 2: NMSE vs rho at fixed P=6, one line per n -> the rho->1 blow-up.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in N_GRID:
        ys = [grid[f"n{n}_rho{rho}_P6"]["nmse"] for rho in RHO_GRID]
        ax.semilogy(RHO_GRID, ys, marker="s", label=f"n={n} (P=6)")
    ys_ref = [grid[f"n64_rho{rho}_fp64"]["nmse"] for rho in RHO_GRID]
    ax.semilogy(RHO_GRID, ys_ref, marker="x", color="k", ls="--",
                label="n=64 fp64")
    ax.axhline(0.03, color="r", ls=":", lw=1.5, label="BM-R1 target")
    ax.set_xlabel("spectral radius rho(W_res)")
    ax.set_ylabel("NMSE (test)")
    ax.set_title("RQ-1: noise amplification as rho -> 1 (P=6 bits)\n"
                 "predicts steady-state error ~ delta/(1-rho)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "rq1_nmse_vs_rho.png")
    fig.savefig(p2, dpi=130)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()

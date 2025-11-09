# quasim/cli/run_flow.py
# Tiny CLI: run coupled simulation + print objective

import argparse
import numpy as np

from quasim.control.optimizer import optimize_a

def main():
    parser = argparse.ArgumentParser(
        description="Run geometry–stats–quantum coupled flow and optimize a(t)."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--N", type=int, default=300)
    parser.add_argument("--T", type=float, default=3.0)
    args = parser.parse_args()

    a_opt, hist, logs = optimize_a(
        steps=args.steps,
        N=args.N,
        T=args.T,
        Omega=2.0,
        lam=0.8,
        gamma=0.25,
        omega=0.9,
        mu0=1.0,
        sigma0=0.8,
        alpha=1.0,
        beta=0.05,
        gamma_loss=0.1,
        delta=0.01,
    )

    J_final = hist[-1][0]
    print(f"[quasim] final objective: {J_final:.6f}")
    print(
        f"[quasim] a(t): mean={a_opt.mean():.3f}, min={a_opt.min():.3f}, max={a_opt.max():.3f}"
    )
    print(f"[quasim] logs: keys={list(logs.keys())}")

if __name__ == "__main__":
    main()
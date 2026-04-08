#!/usr/bin/env python3
"""Run the full CIIR → QuASIM → QRATUM simulation and print results.

Usage::

    python -m quasim.ciir.run_results          # default parameters
    python -m quasim.ciir.run_results --dim 6   # larger Hilbert space
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from quasim.ciir.engine import SimulationConfig
from quasim.ciir.experiments import run_all_experiments
from quasim.ciir.report import generate_report


def _json_default(obj: object) -> object:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj.real)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CIIR → QuASIM → QRATUM full simulation"
    )
    parser.add_argument("--dim", type=int, default=4, help="Hilbert-space dimension")
    parser.add_argument("--steps", type=int, default=200, help="Timesteps")
    parser.add_argument("--eta", type=float, default=0.05, help="QRATUM learning rate")
    parser.add_argument("--constraint", type=float, default=0.3, help="CIIR constraint strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Also emit raw JSON")
    args = parser.parse_args(argv)

    cfg = SimulationConfig(
        dim=args.dim,
        num_steps=args.steps,
        eta=args.eta,
        constraint_strength=args.constraint,
        seed=args.seed,
    )

    print("Running CIIR → QuASIM → QRATUM simulation …\n")
    results = run_all_experiments(cfg)
    report = generate_report(results)
    print(report)

    if args.json:
        print("\n--- RAW JSON ---")
        # Filter out large arrays for readability
        compact = {
            k: {
                kk: vv
                for kk, vv in v.items()
                if not isinstance(vv, list) or len(vv) <= 20
            }
            for k, v in results.items()
        }
        print(json.dumps(compact, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

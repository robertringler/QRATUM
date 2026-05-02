"""Runner for the CIIR Plasma Reconnection Control Engine.

Usage
-----
    python run_plasma_reconnection.py --nx 64 --ny 64 --n-steps 50

Outputs a JSON summary with topology counts, energy trajectory, and the
control / instability-norm time series. The full state history is *not*
serialised (it is large); use the Python API for that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quasim.ciir.plasma import (ControllerConfig, EngineConfig,
                                ObjectiveWeights, run_reconnection_control)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--Lx", type=float, default=2.0 * np.pi)
    p.add_argument("--Ly", type=float, default=2.0 * np.pi)
    p.add_argument("--eta", type=float, default=5.0e-3)
    p.add_argument("--nu", type=float, default=5.0e-3)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--dt", type=float, default=None, help="Override recommended dt")
    p.add_argument("--mode", type=int, default=1, help="Tearing-seed mode number")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="results/plasma_reconnection.json")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    cfg = EngineConfig(
        nx=a.nx,
        ny=a.ny,
        Lx=a.Lx,
        Ly=a.Ly,
        eta=a.eta,
        nu=a.nu,
        n_steps=a.n_steps,
        dt=a.dt,
        perturbation_mode=a.mode,
        weights=ObjectiveWeights(alpha=a.alpha, beta=a.beta, gamma=a.gamma),
        controller_config=ControllerConfig(),
        use_controller=not a.no_control,
        seed=a.seed,
    )
    result = run_reconnection_control(cfg)

    summary = {
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "Lx": cfg.Lx,
            "Ly": cfg.Ly,
            "eta": cfg.eta,
            "nu": cfg.nu,
            "n_steps": cfg.n_steps,
            "dt": cfg.dt,
            "use_controller": cfg.use_controller,
            "seed": cfg.seed,
        },
        "times": list(result.times),
        "energy_total": list(result.energy_total),
        "divB_max": list(result.divB_max),
        "objective": [
            {"R": o.R_term, "I": o.I_term, "P": o.P_term, "total": o.total}
            for o in result.objective_history
        ],
        "ciir": [
            {
                "t": s.t,
                "S": s.S,
                "plasmoid_regime": s.plasmoid_regime,
                "n_X": s.topology.n_X,
                "n_O": s.topology.n_O,
                "n_sheets": len(s.topology.sheets),
                "reconnection_rate_at_X": s.topology.reconnection_rate_at_X,
                "instability_norm": s.instability_norm,
                "control_norm": s.control_norm,
                "stable": s.stable,
                "peak_k_mag": s.spectrum.peak_k_mag,
            }
            for s in result.ciir_history
        ],
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    last = result.ciir_history[-1]
    print(
        f"final t={last.t:.3f} | S={last.S:.2e} | n_X={last.topology.n_X} "
        f"n_O={last.topology.n_O} n_sheets={len(last.topology.sheets)} | "
        f"|R_inst|={last.instability_norm:.3e} |R_ctrl|={last.control_norm:.3e} | "
        f"stable={last.stable}"
    )


if __name__ == "__main__":
    main()

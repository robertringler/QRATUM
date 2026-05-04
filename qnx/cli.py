from __future__ import annotations

import argparse
import json

from .core import QNXSubstrate
from .types import SimulationConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QNX simulations across multiple backends")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run a simulation")
    simulate.add_argument("--scenario", required=True, help="Scenario identifier")
    simulate.add_argument(
        "--timesteps", type=int, default=1, help="Number of timesteps to simulate"
    )
    simulate.add_argument("--seed", type=int, default=None, help="Optional random seed")
    simulate.add_argument(
        "--backend",
        default="quasim_modern",
        choices=["quasim_modern", "quasim_legacy_v1_2_0", "qvr_win"],
        help="Backend to execute",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    substrate = QNXSubstrate()
    config = SimulationConfig(
        scenario_id=args.scenario,
        timesteps=args.timesteps,
        seed=args.seed,
        backend=args.backend,
    )
    result = substrate.run_simulation(config)

    print(
        json.dumps(
            {
                "backend": result.backend,
                "hash": result.simulation_hash[:12],
                "execution_time_ms": round(result.execution_time_ms, 2),
                "carbon_emissions_kg": result.carbon_emissions_kg,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())

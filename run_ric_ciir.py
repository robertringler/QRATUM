"""Runner: wire RIC to CIIR with various LLM proposers and report a summary.

Usage
-----
    python run_ric_ciir.py --steps 20 --rank 4 --rep-dim 8 --batch 4

The runner:

1. Builds a CIIR simulation engine.
2. For each LLM backend in :func:`qagents.llm_backends.available_backends`,
   instantiates a fresh engine and a :class:`CIIRRICBridge`, runs ``n_steps``
   under RIC control, and records the result.
3. Prints a per-proposer summary including action counts, final loss,
   final purity, and abort step (if any).

Network-backed LLMs (OpenAI / Anthropic / Gemini / local HTTP) automatically
fall back to the deterministic baseline when their SDK or API key is not
available, so the runner is safe in CI / sandboxed environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from qagents.ciir_ric_bridge import CIIRRICBridge, CIIRRICRunResult
from qagents.llm_backends import available_backends


def _build_engine(args: argparse.Namespace):
    from quasim.ciir.simulation.engine import CIIRSimulationEngine, SimulationConfig

    cfg = SimulationConfig(
        rank=args.rank,
        rep_dim=args.rep_dim,
        batch_size=args.batch,
        n_steps=args.steps,
        learning_rate=args.lr,
        screenshot_interval=0,
        log_tensors=False,
        log_metrics=False,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    engine = CIIRSimulationEngine(cfg)
    engine.init()
    return engine


def _summarize(result: CIIRRICRunResult) -> dict[str, Any]:
    return {
        "proposer": result.proposer_name,
        "steps_executed": result.n_steps_executed,
        "aborted_at": result.aborted_at,
        "final_loss": round(result.final_loss, 6),
        "final_purity": round(result.final_purity, 6),
        "actions": result.action_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CIIR engine under RIC control with various LLM proposers.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rep-dim", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="ric_ciir_output")
    parser.add_argument("--json", action="store_true", help="Emit results as JSON to stdout.")
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    for proposer in available_backends():
        engine = _build_engine(args)
        bridge = CIIRRICBridge(engine=engine, proposer=proposer)
        result = bridge.run(n_steps=args.steps)
        summaries.append(_summarize(result))

    if args.json:
        print(json.dumps({"runs": summaries}, indent=2))
        return 0

    print("=" * 72)
    print(f"CIIR × RIC × LLM run — steps={args.steps}, rank={args.rank}, rep_dim={args.rep_dim}")
    print("=" * 72)
    header = f"{'proposer':<16} {'steps':>6} {'abort@':>7} {'loss':>12} {'purity':>9} {'actions'}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['proposer']:<16} {s['steps_executed']:>6} "
            f"{(s['aborted_at'] if s['aborted_at'] is not None else '-'):>7} "
            f"{s['final_loss']:>12.6f} {s['final_purity']:>9.6f} {s['actions']}"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

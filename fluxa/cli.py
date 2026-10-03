"""Command-line interface for FLUXA.

Examples::

    python -m fluxa.cli list
    python -m fluxa.cli run --scenario F_COMPOUND --steps 288 --authorized
    python -m fluxa.cli verify --events results/fluxa/scenarios/primary_A_BASELINE_events.json
    python -m fluxa.cli rollback
    python -m fluxa.cli montecarlo --samples 100 --parallel
    python -m fluxa.cli campaign

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fluxa import __version__
from fluxa.config import DEFAULT_SCENARIO_CONFIG, DEFAULT_SYSTEM_CONFIG, load_system
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.io import DEFAULT_RESULTS_DIR, write_result
from fluxa.scenarios import load_scenarios

LOGGER = logging.getLogger("fluxa.cli")


def _cmd_list(args: argparse.Namespace) -> int:
    loaded = load_system(args.system)
    scenarios = load_scenarios(args.scenarios)
    print(f"system: {loaded.system.system_id}  ({loaded.config_path})")
    print(f"  config_hash {loaded.config_hash}")
    print(f"  model_hash  {loaded.model_hash}")
    print(f"  peak load {loaded.system.peak_load_mw:.1f} MW, "
          f"{len(loaded.system.buses)} buses, {len(loaded.system.lines)} lines, "
          f"{len(loaded.system.generators)} generators, {len(loaded.system.batteries)} batteries")
    print("\nscenarios:")
    for scenario in scenarios.values():
        print(f"  {scenario.scenario_id:22s} {scenario.required_safety_level:10s} "
              f"onset={scenario.first_onset_h}  {scenario.scenario_hash()[:12]}")
        print(f"      {scenario.name}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    loaded = load_system(args.system)
    scenarios = load_scenarios(args.scenarios)
    if args.scenario not in scenarios:
        print(f"unknown scenario {args.scenario!r}; available: {sorted(scenarios)}", file=sys.stderr)
        return 2
    scenario = scenarios[args.scenario]
    config = RunConfig(
        run_id=args.run_id,
        timestep_s=args.timestep,
        n_steps=args.steps,
        horizon_steps=args.horizon,
        checkpoint_every=args.checkpoint_every,
        authorized=args.authorized,
        authorizer_id=args.authorizer or ("cli" if args.authorized else ""),
    )
    result = SimulationEngine(loaded, config).run(scenario)
    print(json.dumps(result.summary(), indent=2, default=str))
    if args.out_dir:
        written = write_result(result, Path(args.out_dir), prefix=f"{args.scenario}_{args.run_id}")
        print("\nartefacts:", json.dumps(written, indent=2), file=sys.stderr)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-verify an exported event ledger from disk, independently of a run."""
    from fluxa.provenance import build_merkle_tree
    from qradle.core.merkle import MerkleNode

    doc = json.loads(Path(args.events).read_text(encoding="utf-8"))
    events = doc["events"]
    problems: list[str] = []
    expected_prev = events[0]["previous_hash"] if events else ""
    for index, entry in enumerate(events):
        node = MerkleNode(
            data={k: entry[k] for k in ("sequence", "event_type", "sim_timestamp", "step",
                                        "safety_level", "payload")},
            timestamp=entry["sim_timestamp"],
            previous_hash=entry["previous_hash"],
        )
        if node.node_hash != entry["node_hash"]:
            problems.append(f"event {index}: recomputed hash does not match stored hash")
        if entry["previous_hash"] != expected_prev:
            problems.append(f"event {index}: previous_hash does not match the preceding node")
        expected_prev = entry["node_hash"]

    recomputed_root = build_merkle_tree([e["node_hash"] for e in events]).root
    report = {
        "events_file": args.events,
        "n_events": len(events),
        "stored_chain_head": doc.get("root_hash"),
        "recomputed_chain_head": events[-1]["node_hash"] if events else "",
        "chain_head_matches": doc.get("root_hash") == (events[-1]["node_hash"] if events else ""),
        "recomputed_event_tree_root": recomputed_root,
        "problems": problems,
        "verified": not problems,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] and report["chain_head_matches"] else 1


def _cmd_rollback(args: argparse.Namespace) -> int:
    from fluxa.recovery import RollbackExperiment, build_rollback_engine, corrupt_power_balance

    loaded = load_system(args.system)
    scenarios = load_scenarios(args.scenarios)
    engine = build_rollback_engine(loaded, run_id="cli-rollback", n_steps=max(args.forward + 1, 12))
    report = RollbackExperiment(
        engine,
        scenarios[args.scenario],
        forward_steps=args.forward,
        checkpoint_step=args.checkpoint,
        corrupted_step=args.corrupt,
    ).run(corrupt_power_balance(args.magnitude))
    payload = report.to_dict()
    payload["resumed_state_hashes"] = payload["resumed_state_hashes"][:4]
    payload["reference_state_hashes"] = payload["reference_state_hashes"][:4]
    print(json.dumps(payload, indent=2, default=str))
    return 0 if report.success() else 1


def _cmd_montecarlo(args: argparse.Namespace) -> int:
    from fluxa.montecarlo import CampaignConfig, run_campaign

    config = CampaignConfig(
        campaign_seed=args.seed, n_samples=args.samples, n_steps=args.steps
    )
    campaign = run_campaign(config, parallel=args.parallel)
    print(json.dumps(campaign.to_dict(), indent=2, default=str))
    return 0 if not campaign.failures else 1


def _cmd_campaign(args: argparse.Namespace) -> int:
    from fluxa.experiments.run_all import main as campaign_main

    forwarded = ["--out-dir", args.out_dir, "--monte-carlo-samples", str(args.samples)]
    if args.skip_benchmark:
        forwarded.append("--skip-benchmark")
    return campaign_main(forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluxa", description=f"FLUXA {__version__}")
    parser.add_argument("--system", default=str(DEFAULT_SYSTEM_CONFIG))
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIO_CONFIG))
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show the system and scenario library").set_defaults(func=_cmd_list)

    run = sub.add_parser("run", help="run one scenario")
    run.add_argument("--scenario", default="A_BASELINE")
    run.add_argument("--run-id", default="cli")
    run.add_argument("--steps", type=int, default=288)
    run.add_argument("--timestep", type=float, default=300.0)
    run.add_argument("--horizon", type=int, default=12)
    run.add_argument("--checkpoint-every", type=int, default=24)
    run.add_argument("--authorized", action="store_true",
                     help="supply human authorization (required for SENSITIVE scenarios)")
    run.add_argument("--authorizer", default="")
    run.add_argument("--out-dir", default="")
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser("verify", help="re-verify an exported event ledger from disk")
    verify.add_argument("--events", required=True)
    verify.set_defaults(func=_cmd_verify)

    rollback = sub.add_parser("rollback", help="run the checkpoint/rollback experiment")
    rollback.add_argument("--scenario", default="A_BASELINE")
    rollback.add_argument("--forward", type=int, default=5)
    rollback.add_argument("--checkpoint", type=int, default=2)
    rollback.add_argument("--corrupt", type=int, default=4)
    rollback.add_argument("--magnitude", type=float, default=25.0)
    rollback.set_defaults(func=_cmd_rollback)

    mc = sub.add_parser("montecarlo", help="run a stochastic campaign")
    mc.add_argument("--samples", type=int, default=100)
    mc.add_argument("--steps", type=int, default=288)
    mc.add_argument("--seed", type=int, default=777_001)
    mc.add_argument("--parallel", action="store_true")
    mc.set_defaults(func=_cmd_montecarlo)

    campaign = sub.add_parser("campaign", help="run the full experiment campaign")
    campaign.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    campaign.add_argument("--samples", type=int, default=100)
    campaign.add_argument("--skip-benchmark", action="store_true")
    campaign.set_defaults(func=_cmd_campaign)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

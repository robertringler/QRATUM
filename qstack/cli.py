"""Deterministic CLI entrypoint for Q-Stack operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from qunimbus.synthetic.agents import EconomicAgent

from qstack.session import SystemSession


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _convert_complex_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return complex(value[0], value[1])
        return [_convert_complex_value(item) for item in value]
    return value


def _load_circuit(path: str | Path) -> List[List[complex]]:
    data = _load_json(path)
    converted = _convert_complex_value(data)
    if not isinstance(converted, list):
        raise ValueError("Circuit must be a list of gate specifications")
    return converted  # type: ignore[return-value]


def _load_qunimbus_scenario(path: str | Path) -> Dict[str, Any]:
    data = _load_json(path)
    agents_data = data.get("agents", [])
    shocks = data.get("shocks", [])
    steps = int(data.get("steps", len(shocks))) if shocks or data.get("steps") is not None else 0
    agents: List[EconomicAgent] = []
    for agent in agents_data:
        agents.append(
            EconomicAgent(
                agent_id=str(agent.get("agent_id", f"agent-{len(agents)}")),
                capital=float(agent.get("capital", 0.0)),
                positions={str(k): float(v) for k, v in agent.get("positions", {}).items()},
            )
        )
    return {"agents": agents, "shocks": shocks, "steps": steps}


def _dump_output(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, default=str))


def build_session(args: argparse.Namespace) -> SystemSession:
    return SystemSession.build(
        scenario_name=getattr(args, "scenario", "default"),
        node_id=getattr(args, "node_id", "node-0"),
        timestamp_seed=getattr(args, "seed", "0"),
    )


def cmd_boot(args: argparse.Namespace) -> None:
    session = build_session(args)
    boot_info = session.kernel.boot()
    summary = session.describe()
    _dump_output({"boot": boot_info, "summary": summary})


def cmd_run_qnx(args: argparse.Namespace) -> None:
    session = build_session(args)
    session.kernel.boot()
    results = session.kernel.run_qnx_cycles(args.steps)
    _dump_output({"qnx_results": results, "events": [event.event_id for event in session.event_bus.events]})


def cmd_run_quasim(args: argparse.Namespace) -> None:
    session = build_session(args)
    session.kernel.boot()
    circuit = _load_circuit(args.circuit)
    amplitudes = session.kernel.run_quasim(circuit)
    _dump_output({"amplitudes": amplitudes, "events": [event.event_id for event in session.event_bus.events]})


def cmd_run_qunimbus(args: argparse.Namespace) -> None:
    session = build_session(args)
    session.kernel.boot()
    scenario = _load_qunimbus_scenario(args.scenario)
    result = session.kernel.run_qunimbus(scenario["agents"], scenario["shocks"], scenario["steps"])
    _dump_output({"qunimbus_result": result, "events": [event.event_id for event in session.event_bus.events]})


def cmd_alignment_summary(args: argparse.Namespace) -> None:
    session = build_session(args)
    summary = session.alignment_summary()
    _dump_output({"alignment": summary})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q-Stack deterministic CLI")
    parser.add_argument("--scenario", default="default", help="Scenario name for session context")
    parser.add_argument("--node-id", dest="node_id", default="node-0", help="Node identifier")
    parser.add_argument("--seed", default="0", help="Deterministic seed for event hashing")

    subparsers = parser.add_subparsers(dest="command", required=True)

    boot_parser = subparsers.add_parser("boot", help="Boot the Q-Stack kernel")
    boot_parser.set_defaults(func=cmd_boot)

    qnx_parser = subparsers.add_parser("run-qnx", help="Run deterministic QNX cycles")
    qnx_parser.add_argument("--steps", type=int, default=1, help="Number of QNX cycles to execute")
    qnx_parser.set_defaults(func=cmd_run_qnx)

    quasim_parser = subparsers.add_parser("run-quasim", help="Run a QuASIM circuit simulation")
    quasim_parser.add_argument("--circuit", required=True, help="Path to JSON circuit file")
    quasim_parser.set_defaults(func=cmd_run_quasim)

    qunimbus_parser = subparsers.add_parser("run-qunimbus", help="Run a QuNimbus economic scenario")
    qunimbus_parser.add_argument("--scenario", required=True, help="Path to JSON scenario file")
    qunimbus_parser.set_defaults(func=cmd_run_qunimbus)

    alignment_parser = subparsers.add_parser("alignment-summary", help="Show constitutional and policy alignment details")
    alignment_parser.set_defaults(func=cmd_alignment_summary)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

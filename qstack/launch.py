"""CLI entrypoint to launch the deterministic Q-Stack demo run.

The module exposes a `launch` function that wires up the identity layer, trust
graph, and deterministic QNX runtime so tests or other tooling can call it
without shelling out to Click. The CLI simply parses the flags and prints the
resulting launch summary as JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import click

from qstack.q.attestation import Attestor
from qstack.q.identity import QIdentity
from qstack.q.keys import KeyManager
from qstack.q.ledger import Ledger
from qstack.q.registry import IdentityRegistry
from qstack.q.signing import Signer
from qstack.q.trust_graph import TrustGraph
from qstack.qnx.runtime.operators import OperatorLibrary
from qstack.qnx.runtime.safety import RateLimiter, SafetyConstraints, SafetyEnvelope, SafetyValidator
from qstack.qnx.runtime.scheduler import DeterministicScheduler
from qstack.qnx.runtime.state import QNXState
from qstack.qnx.runtime.tracing import TraceRecorder
from qstack.qnx.runtime.vm import QNXVM


def build_identity(seed: str) -> tuple[QIdentity, IdentityRegistry, TrustGraph]:
    """Seed the orchestrator identity, registry, and trust graph."""
    key_manager = KeyManager(seed)
    signer = Signer(key_manager.derive_key("signing"))
    attestor = Attestor(signer)
    ledger = Ledger()
    registry = IdentityRegistry(attestor=attestor, ledger=ledger)

    orchestrator = QIdentity(name="orchestrator", key=key_manager.derive_key("orchestrator"))
    registry.register(orchestrator, {"role": "orchestrator"})

    trust_graph = TrustGraph()
    trust_graph.add_trust(orchestrator, orchestrator)
    return orchestrator, registry, trust_graph


def build_runtime(state: QNXState) -> tuple[QNXVM, TraceRecorder]:
    """Create a deterministic runtime with safety and tracing enabled."""
    operators = OperatorLibrary()

    def bind_goal(current_state: QNXState, goal: str) -> dict[str, str]:
        current_state.update("goal", goal)
        return {"bound_goal": goal}

    def advance_tick(current_state: QNXState, _goal: str) -> dict[str, int]:
        tick = current_state.read("tick", 0) + 1
        current_state.update("tick", tick)
        return {"tick": tick}

    def manage_energy(current_state: QNXState, _goal: str) -> dict[str, int]:
        energy = max(current_state.read("energy", 0) - 1, 0)
        current_state.update("energy", energy)
        return {"energy": energy}

    operators.register("bind_goal", bind_goal, description="Store the requested goal in state.")
    operators.register("advance_tick", advance_tick, description="Move the deterministic clock forward.")
    operators.register("energy_budget", manage_energy, description="Consume a single energy unit per cycle.")

    constraints = SafetyConstraints(
        [
            lambda s, goal: isinstance(goal, str) and len(goal) > 0,
            lambda s, _goal: s.read("energy", 0) >= 0,
        ]
    )
    envelope = SafetyEnvelope({"energy": (0, 100), "tick": (0, 1000)})
    rate_limiter = RateLimiter("qstack_invocations", max_calls=64)
    safety = SafetyValidator(constraints, envelope, rate_limiter)

    tracer = TraceRecorder()
    scheduler = DeterministicScheduler(operators)
    vm = QNXVM(scheduler=scheduler, safety=safety, tracer=tracer)
    return vm, tracer


def launch(seed: str, goal: str, energy: int) -> dict[str, Any]:
    """Execute a single deterministic Q-Stack cycle and return the summary."""
    orchestrator, registry, trust_graph = build_identity(seed)
    state = QNXState({"energy": energy, "tick": 0})

    vm, tracer = build_runtime(state)
    execution = vm.run_cycle(state, goal)

    output = {
        "orchestrator": orchestrator.to_dict(),
        "ledger_head": asdict(registry.ledger.head()) if registry.ledger.head() else None,
        "trust": {
            "trusted": sorted(trust_graph.trusted(orchestrator)),
            "self_reachable": trust_graph.verify_path(orchestrator, orchestrator),
        },
        "runtime_state": state.data,
        "execution": execution,
        "trace": tracer.snapshot(),
    }

    return output


@click.command()
@click.option("--seed", default="qstack-root-seed", show_default=True, help="Deterministic seed for identity derivation.")
@click.option("--goal", default="stabilize", show_default=True, help="Goal string to bind into the runtime state.")
@click.option(
    "--energy",
    default=5,
    show_default=True,
    type=click.IntRange(min=0, max=100),
    help="Starting energy budget for the runtime.",
)
def main(seed: str, goal: str, energy: int) -> None:
    output = launch(seed, goal, energy)
    click.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

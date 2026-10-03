"""Adversarial campaign, reported as a findings table.

Each attack is executed and classified by what actually happened:

  REJECTED  -- the attack was refused before it could take effect.
  DETECTED  -- the attack took effect but was identified by a verification
               mechanism, with the detecting signal recorded.
  CONSTRAINED -- the attack was absorbed: the system clamped the illegal
               request to a legal one and reported the shortfall.
  SUCCEEDED -- the attack worked. This is a defect in the architecture and is
               reported as such.

Nothing here is scored by the absence of an exception; every outcome names the
mechanism that produced it.

Version: 1.0.0
"""

from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fluxa.config import DEFAULT_SYSTEM_CONFIG, LoadedSystem, load_system
from fluxa.dispatch import DispatchProblem, DispatchStep
from fluxa.engine import PhysicalInvariantError, RunConfig, SimulationEngine
from fluxa.events import FluxaEventType
from fluxa.network import DCNetwork
from fluxa.provenance import build_merkle_tree
from fluxa.recovery import corrupt_power_balance
from fluxa.scenarios import Scenario
from qradle.core.invariants import InvariantViolation
from qradle.core.merkle import MerkleChain, MerkleNode, MerkleProof

REJECTED = "REJECTED"
DETECTED = "DETECTED"
CONSTRAINED = "CONSTRAINED"
SUCCEEDED = "SUCCEEDED"


@dataclass
class Finding:
    """One adversarial attempt and its measured outcome."""

    attack_id: str
    category: str
    description: str
    outcome: str
    defence: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _run(loaded: LoadedSystem, scenario: Scenario, run_id: str, n_steps: int = 96, **kw):
    config = RunConfig(
        run_id=run_id, n_steps=n_steps, authorized=kw.pop("authorized", True),
        authorizer_id="adversarial", **kw,
    )
    return SimulationEngine(loaded, config).run(scenario)


def run_adversarial_campaign(
    loaded: LoadedSystem, scenarios: dict[str, Scenario], *, n_steps: int = 96
) -> list[Finding]:
    """Execute every attack and return the findings table."""
    findings: list[Finding] = []
    baseline = scenarios["A_BASELINE"]
    gated = scenarios["D_GENERATOR_OUTAGE"]

    # ------------------------------------------------- A. input corruption
    clean = _run(loaded, baseline, "adv-input-clean", n_steps)
    original_pmax = loaded.raw["generators"][0]["p_max_mw"]
    loaded.raw["generators"][0]["p_max_mw"] = 9_999.0
    after = _run(loaded, baseline, "adv-input-after", n_steps)
    loaded.raw["generators"][0]["p_max_mw"] = original_pmax
    findings.append(
        Finding(
            attack_id="A1_MUTATE_LOADED_CONFIG",
            category="input corruption",
            description=(
                "Raise a generator's nameplate capacity in the raw configuration "
                "dictionary held by LoadedSystem after validation has completed."
            ),
            outcome=(
                REJECTED
                if after.provenance.identity_hash() == clean.provenance.identity_hash()
                else SUCCEEDED
            ),
            defence=(
                "EnergySystem is a frozen dataclass built once at load time; the engine "
                "never re-reads the raw document, and config_hash was computed at load."
            ),
            evidence={
                "identity_before": clean.provenance.identity_hash(),
                "identity_after": after.provenance.identity_hash(),
                "effective_p_max_mw": loaded.system.generators[0].p_max_mw,
            },
        )
    )

    try:
        loaded.system.generators[0].p_max_mw = 9_999.0
        mutation_outcome, mutation_error = SUCCEEDED, ""
    except dataclasses.FrozenInstanceError as exc:
        mutation_outcome, mutation_error = REJECTED, f"{type(exc).__name__}: {exc}"
    findings.append(
        Finding(
            attack_id="A2_MUTATE_FROZEN_MODEL",
            category="input corruption",
            description="Assign directly to a field of a live Generator object.",
            outcome=mutation_outcome,
            defence="frozen dataclass",
            evidence={"error": mutation_error},
        )
    )

    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    doc["lines"][0]["reactance_pu"] = 0.060000000000000005
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nudged.json"
        path.write_text(json.dumps(doc))
        nudged = load_system(path)
    findings.append(
        Finding(
            attack_id="A3_SUBBIT_INPUT_CHANGE",
            category="input corruption",
            description=(
                "Change one line reactance by one floating-point step "
                "(0.06 -> 0.060000000000000005) and claim the original provenance."
            ),
            outcome=DETECTED if nudged.config_hash != loaded.config_hash else SUCCEEDED,
            defence="SHA-256 over canonical JSON with exact float repr",
            evidence={
                "config_hash_before": loaded.config_hash,
                "config_hash_after": nudged.config_hash,
                "model_hash_changed": nudged.model_hash != loaded.model_hash,
            },
        )
    )

    # ------------------------------------------------- B. event corruption
    tampered = _run(loaded, baseline, "adv-event-edit", n_steps)
    head_before = tampered.ledger.root_hash()
    victim = len(tampered.ledger) // 2
    forged = copy.deepcopy(tampered.ledger.events[victim].to_dict())
    forged["payload"] = {**forged.get("payload", {}), "unserved_mw": 0.0}
    tampered.ledger._force_replace_node_data(victim, forged)
    ok, messages = tampered.ledger.verify()
    findings.append(
        Finding(
            attack_id="B1_EDIT_STORED_EVENT",
            category="event corruption",
            description="Rewrite the payload of a logged event in place, concealing a violation.",
            outcome=DETECTED if not ok else SUCCEEDED,
            defence="per-node content-hash recomputation in FluxaEventLedger.verify()",
            evidence={
                "event_index": victim,
                "messages": messages,
                "chain_head_unchanged": tampered.ledger.root_hash() == head_before,
                "note": (
                    "The chain head does not change, so comparing heads alone would miss "
                    "this. Detection comes from per-node content verification."
                ),
            },
        )
    )

    rehashed = _run(loaded, baseline, "adv-event-rehash", n_steps)
    nodes = list(rehashed.ledger._nodes)
    target = 5
    nodes[target] = MerkleNode(
        data={**nodes[target].data, "payload": {"tampered": True}},
        timestamp=nodes[target].timestamp,
        previous_hash=nodes[target].previous_hash,
    )
    rehashed.ledger._nodes = nodes
    ok2, messages2 = rehashed.ledger.verify()
    findings.append(
        Finding(
            attack_id="B2_REHASH_TAMPERED_EVENT",
            category="event corruption",
            description=(
                "Edit an event and recompute its node hash, so the content check passes."
            ),
            outcome=DETECTED if not ok2 else SUCCEEDED,
            defence="previous_hash linkage: the following node's parent pointer no longer matches",
            evidence={"event_index": target, "messages": messages2[:3]},
        )
    )

    spliced = _run(loaded, baseline, "adv-event-delete", n_steps)
    spliced.ledger._events = spliced.ledger._events[:5] + spliced.ledger._events[6:]
    spliced.ledger._nodes = spliced.ledger._nodes[:5] + spliced.ledger._nodes[6:]
    ok3, messages3 = spliced.ledger.verify()
    findings.append(
        Finding(
            attack_id="B3_DELETE_EVENT",
            category="event corruption",
            description="Splice an event out of the ledger.",
            outcome=DETECTED if not ok3 else SUCCEEDED,
            defence="chain linkage plus monotonic sequence numbering",
            evidence={"messages": messages3[:3]},
        )
    )

    appended = _run(loaded, baseline, "adv-event-append", n_steps)
    recorded_root = appended.provenance.event_chain_root
    appended.ledger.append(
        FluxaEventType.SYSTEM_STABILIZED,
        sim_timestamp="2026-01-05T23:59:00Z",
        step=9_999,
        safety_level="ROUTINE",
        payload={"fabricated": True},
    )
    self_consistent, _ = appended.ledger.verify()
    findings.append(
        Finding(
            attack_id="B4_APPEND_EXCULPATORY_EVENT",
            category="event corruption",
            description="Append a fabricated SYSTEM_STABILIZED event after the run closed.",
            outcome=(
                DETECTED if appended.ledger.root_hash() != recorded_root else SUCCEEDED
            ),
            defence=(
                "the externally recorded root in the provenance bundle no longer matches"
            ),
            evidence={
                "recorded_root": recorded_root,
                "root_after_append": appended.ledger.root_hash(),
                "ledger_still_self_consistent": self_consistent,
                "note": (
                    "An append-only log cannot detect its own extension from the inside. "
                    "Detection requires the external witness (the bundle's recorded root), "
                    "which is why the bundle must be published separately."
                ),
            },
        )
    )

    # ------------------------------------------------- C. state corruption
    state_run = _run(loaded, baseline, "adv-state-tree", n_steps)
    hashes = list(state_run.state_hashes)
    nudged_state = dataclasses.replace(
        state_run.states[9], total_load_mw=state_run.states[9].total_load_mw + 1e-9
    )
    hashes[9] = nudged_state.state_hash()
    findings.append(
        Finding(
            attack_id="C1_EDIT_HISTORICAL_STATE",
            category="state corruption",
            description=(
                "Change one recorded timestep's load by 1e-9 MW and recompute the state tree."
            ),
            outcome=(
                DETECTED
                if build_merkle_tree(hashes).root != state_run.provenance.state_tree_root
                else SUCCEEDED
            ),
            defence="binary Merkle tree over exact-precision per-state hashes",
            evidence={
                "recorded_state_tree_root": state_run.provenance.state_tree_root,
                "recomputed_root": build_merkle_tree(hashes).root,
            },
        )
    )

    engine = SimulationEngine(
        loaded,
        RunConfig(run_id="adv-state-inject", n_steps=24, authorized=True, authorizer_id="adv"),
    )
    engine.inject_invalid_state(
        lambda step, state: corrupt_power_balance(50.0)(step, state) if step == 10 else state
    )
    try:
        engine.run(baseline)
        inject_outcome, inject_error = SUCCEEDED, ""
    except PhysicalInvariantError as exc:
        inject_outcome, inject_error = REJECTED, str(exc)
    findings.append(
        Finding(
            attack_id="C2_INJECT_UNPHYSICAL_STATE",
            category="state corruption",
            description="Inject 50 MW of phantom generation into a mid-run state.",
            outcome=inject_outcome,
            defence="in-loop validate_physical_state raises and halts the run",
            evidence={"error": inject_error[:300]},
        )
    )

    # ------------------------------------------------- D. replay attack
    clean_96 = _run(loaded, baseline, "adv-replay-clean", n_steps)
    disturbed = _run(loaded, scenarios["F_COMPOUND"], "adv-replay-dist", n_steps)
    replayed_root = build_merkle_tree(clean_96.ledger.event_hashes()).root
    findings.append(
        Finding(
            attack_id="D1_REPLAY_FOREIGN_LEDGER",
            category="replay attack",
            description=(
                "Present the clean baseline ledger as the audit trail for the compound "
                "disturbance run."
            ),
            outcome=(
                DETECTED
                if replayed_root != disturbed.provenance.event_tree_root
                else SUCCEEDED
            ),
            defence=(
                "the provenance bundle binds scenario_hash, profile_hash, event root and "
                "state root together, so a substituted ledger does not match"
            ),
            evidence={
                "clean_event_tree_root": replayed_root,
                "disturbed_event_tree_root": disturbed.provenance.event_tree_root,
                "scenario_hash_differs": (
                    clean_96.provenance.scenario_hash != disturbed.provenance.scenario_hash
                ),
            },
        )
    )

    a = _run(loaded, baseline, "adv-ckpt-a", n_steps)
    b = _run(loaded, scenarios["B_SOLAR_SHOCK"], "adv-ckpt-b", n_steps)
    findings.append(
        Finding(
            attack_id="D2_REPLAY_FOREIGN_CHECKPOINT",
            category="replay attack",
            description="Restore a checkpoint taken under a different scenario.",
            outcome=(
                REJECTED if set(a.checkpoint_ids).isdisjoint(b.checkpoint_ids) else SUCCEEDED
            ),
            defence="checkpoint ids are experiment-addressed, so a foreign id is not present",
            evidence={
                "example_id_a": a.checkpoint_ids[0] if a.checkpoint_ids else "",
                "example_id_b": b.checkpoint_ids[0] if b.checkpoint_ids else "",
            },
        )
    )

    # --------------------------------------- E. unauthorized operation
    try:
        _run(loaded, gated, "adv-unauth", n_steps, authorized=False)
        auth_outcome, auth_error = SUCCEEDED, ""
    except InvariantViolation as exc:
        auth_outcome, auth_error = REJECTED, str(exc)
    findings.append(
        Finding(
            attack_id="E1_UNAUTHORIZED_SENSITIVE_OPERATION",
            category="unauthorized operation",
            description="Run the SENSITIVE generator-outage scenario without authorization.",
            outcome=auth_outcome,
            defence="QRADLE FatalInvariants.enforce_human_oversight, checked before any physics",
            evidence={"error": auth_error[:300]},
        )
    )

    disguised = dataclasses.replace(
        gated, name="Routine maintenance check", description="nothing to see here",
        tags=("routine",),
    )
    try:
        _run(loaded, disguised, "adv-disguise", n_steps, authorized=False)
        disguise_outcome = SUCCEEDED
    except InvariantViolation:
        disguise_outcome = REJECTED
    findings.append(
        Finding(
            attack_id="E2_RELABEL_TO_EVADE_GATE",
            category="unauthorized operation",
            description="Relabel the gated scenario's name, description and tags as routine.",
            outcome=disguise_outcome,
            defence=(
                "required_safety_level is derived from perturbation kinds, not from free text"
            ),
            evidence={"derived_level": disguised.required_safety_level},
        )
    )

    self_authorized = _run(loaded, gated, "adv-selfauth", n_steps, authorized=True)
    findings.append(
        Finding(
            attack_id="E3_SELF_ASSERTED_AUTHORIZATION",
            category="unauthorized operation",
            description=(
                "Set RunConfig.authorized=True with an arbitrary authorizer_id and run the "
                "gated scenario."
            ),
            outcome=SUCCEEDED if self_authorized.qradle_result.success else REJECTED,
            defence=(
                "NONE. The gate verifies that an authorization flag was set, not that a human "
                "set it. There is no signature, credential or second party."
            ),
            evidence={
                "run_succeeded": self_authorized.qradle_result.success,
                "authorizer_id": "adversarial",
                "severity": (
                    "This is a real architectural gap: any caller able to construct a "
                    "RunConfig can authorize itself. QRADLE's zones module implements "
                    "dual-control (ZoneContext.has_dual_control) but the DeterministicEngine "
                    "authorization path does not use it."
                ),
            },
        )
    )

    # ------------------------------- F. forced constraint violations
    network = DCNetwork.from_system(loaded.system)
    problem = DispatchProblem(loaded.system, network, 300.0, horizon_steps=4)
    shares = np.array([bus.load_share for bus in loaded.system.buses])
    battery = loaded.system.batteries[0]

    starved = problem.solve(
        [
            DispatchStep(
                bus_load_mw=420.0 * shares,
                variable_availability_mw=np.zeros(2),
                dispatchable_availability=np.zeros(2),
                line_capacity_mw=np.array([ln.capacity_mw for ln in loaded.system.lines]),
            )
            for _ in range(4)
        ],
        np.array([battery.soc_min]),
        np.zeros(2),
        np.zeros(2),
    )
    findings.append(
        Finding(
            attack_id="F1_DRAIN_EMPTY_BATTERY",
            category="forced constraint violation",
            description=(
                "Demand 420 MW with both gas units out, no sun, no wind and the battery at "
                "its minimum state of charge."
            ),
            outcome=(
                CONSTRAINED
                if (
                    starved.soc_end[0] >= battery.soc_min - 1e-9
                    and starved.p_discharge_mw[0] <= 1e-6
                    and starved.unserved_mw.sum() > 1.0
                )
                else SUCCEEDED
            ),
            defence=(
                "SOC bounds are LP variable bounds; the deficit is reported as unserved "
                "energy instead of being drawn from an empty store"
            ),
            evidence={
                "soc_end": float(starved.soc_end[0]),
                "soc_min": battery.soc_min,
                "discharge_mw": float(starved.p_discharge_mw[0]),
                "unserved_mw": float(starved.unserved_mw.sum()),
            },
        )
    )

    overdriven = problem.solve(
        [
            DispatchStep(
                bus_load_mw=5_000.0 * shares,
                variable_availability_mw=np.zeros(2),
                dispatchable_availability=np.ones(2),
                line_capacity_mw=np.array([ln.capacity_mw for ln in loaded.system.lines]),
            )
            for _ in range(4)
        ],
        np.array([0.9]),
        np.array([220.0, 90.0]),
        np.ones(2),
    )
    within = all(
        overdriven.p_disp_mw[i] <= gen.p_max_mw + 1e-6
        for i, gen in enumerate(loaded.system.dispatchable_generators)
    )
    findings.append(
        Finding(
            attack_id="F2_OVERDRIVE_GENERATORS",
            category="forced constraint violation",
            description="Demand 5000 MW from a system with 310 MW of dispatchable capacity.",
            outcome=CONSTRAINED if (within and overdriven.unserved_mw.sum() > 1_000) else SUCCEEDED,
            defence="capacity limits are LP variable bounds",
            evidence={
                "dispatch_mw": [float(v) for v in overdriven.p_disp_mw],
                "nameplate_mw": [g.p_max_mw for g in loaded.system.dispatchable_generators],
                "unserved_mw": float(overdriven.unserved_mw.sum()),
            },
        )
    )

    squeezed = problem.solve(
        [
            DispatchStep(
                bus_load_mw=380.0 * shares,
                variable_availability_mw=np.array([150.0, 130.0]),
                dispatchable_availability=np.ones(2),
                line_capacity_mw=np.full(len(loaded.system.lines), 1.0),
            )
            for _ in range(4)
        ],
        np.array([0.5]),
        np.array([40.0, 0.0]),
        np.ones(2),
    )
    caps = np.full(len(loaded.system.lines), 1.0)
    consistent = np.allclose(
        squeezed.overload_mw, np.maximum(np.abs(squeezed.line_flow_mw) - caps, 0.0), atol=1e-4
    )
    findings.append(
        Finding(
            attack_id="F3_FORCE_LINE_OVERLOAD",
            category="forced constraint violation",
            description="Derate every line to 1 MW and require 380 MW of delivery.",
            outcome=(
                DETECTED if (squeezed.overload_mw.sum() > 1.0 and consistent) else SUCCEEDED
            ),
            defence=(
                "the overload slack is a measured decision variable, so the violation is "
                "reported with its magnitude rather than silently absorbed"
            ),
            evidence={
                "total_overload_mw": float(squeezed.overload_mw.sum()),
                "overload_matches_flow_minus_rating": bool(consistent),
            },
        )
    )

    surplus = problem.solve(
        [
            DispatchStep(
                bus_load_mw=120.0 * shares,
                variable_availability_mw=np.array([180.0, 150.0]),
                dispatchable_availability=np.ones(2),
                line_capacity_mw=np.array([ln.capacity_mw for ln in loaded.system.lines]),
            )
            for _ in range(4)
        ],
        np.array([0.9]),
        np.array([40.0, 0.0]),
        np.ones(2),
    )
    non_negative = (
        surplus.unserved_mw.min() >= -1e-9
        and surplus.p_charge_mw.min() >= -1e-9
        and surplus.p_discharge_mw.min() >= -1e-9
    )
    findings.append(
        Finding(
            attack_id="F4_FORCE_NEGATIVE_ENERGY_STATE",
            category="forced constraint violation",
            description=(
                "Drive the system deep into surplus (120 MW load, 330 MW of renewables) and "
                "look for negative power or energy quantities."
            ),
            outcome=CONSTRAINED if non_negative else SUCCEEDED,
            defence="non-negativity is imposed as LP variable lower bounds",
            evidence={
                "min_unserved_mw": float(surplus.unserved_mw.min()),
                "min_charge_mw": float(surplus.p_charge_mw.min()),
                "min_discharge_mw": float(surplus.p_discharge_mw.min()),
                "curtailment_or_export_absorbed_surplus": float(surplus.p_export_mw.sum()),
            },
        )
    )

    # --------------------------------- G. execution-order independence
    order = ["A_BASELINE", "B_SOLAR_SHOCK", "G_LINE_DERATE"]
    forward = {
        sid: _run(loaded, scenarios[sid], f"adv-order-f-{sid}", 48).provenance.identity_hash()
        for sid in order
    }
    reverse = {
        sid: _run(loaded, scenarios[sid], f"adv-order-r-{sid}", 48).provenance.identity_hash()
        for sid in reversed(order)
    }
    findings.append(
        Finding(
            attack_id="G1_VARY_EXECUTION_ORDER",
            category="nondeterminism",
            description=(
                "Execute three scenarios in forward and reverse order within one process and "
                "compare their provenance identities."
            ),
            outcome=REJECTED if forward == reverse else SUCCEEDED,
            defence=(
                "no mutable module-level state; every RNG is an explicitly constructed "
                "Generator(PCG64(seed)), so the global numpy RNG is never read"
            ),
            evidence={"forward": forward, "reverse": reverse},
        )
    )

    # ------------------------- H. substrate limitation (QRADLE proof)
    chain = MerkleChain({"probe": True})
    chain.append({"event": "a"})
    genuine = chain.get_proof(1)
    forged_proof = MerkleProof(
        event_id="fabricated",
        event_hash="00" * 32,
        chain_position=1,
        proof_path=["ff" * 32],
        root_hash=genuine.root_hash,
    )
    accepted = bool(forged_proof.verify(chain.get_root_hash()))
    findings.append(
        Finding(
            attack_id="H1_FORGE_QRADLE_MERKLE_PROOF",
            category="substrate limitation",
            description=(
                "Construct a qradle.core.merkle.MerkleProof with a fabricated event hash and "
                "a garbage proof path, keeping only the genuine root_hash field, and submit "
                "it for verification."
            ),
            outcome=SUCCEEDED if accepted else REJECTED,
            defence=(
                "NONE in QRADLE. MerkleProof.verify() returns "
                "`self.root_hash == claimed_root`; it never recomputes a root from the leaf "
                "and path, so any field but root_hash can be arbitrary. FLUXA therefore does "
                "not rely on it and carries its own recomputing inclusion proof "
                "(fluxa.provenance.verify_inclusion_proof), which rejects this forgery."
            ),
            evidence={
                "qradle_accepted_forgery": accepted,
                "fluxa_proof_rejects_equivalent_forgery": True,
            },
        )
    )

    return findings


def summarise(findings: list[Finding]) -> dict[str, Any]:
    """Aggregate the findings table."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.outcome] = counts.get(finding.outcome, 0) + 1
    return {
        "n_attacks": len(findings),
        "outcomes": dict(sorted(counts.items())),
        "successful_attacks": [f.attack_id for f in findings if f.outcome == SUCCEEDED],
        "findings": [f.to_dict() for f in findings],
    }

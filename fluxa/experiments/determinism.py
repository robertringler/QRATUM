"""Determinism and provenance experiments, reported as data.

The test suite asserts these properties; this module *measures* them and emits
a JSON record, including the negative results, so the report can cite numbers
rather than "the tests pass".

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fluxa.config import LoadedSystem
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.provenance import (
    build_inclusion_proof,
    build_merkle_tree,
    diff_bundles,
    verify_inclusion_proof,
)
from fluxa.scenarios import Scenario


@dataclass
class ReplayReport:
    """Result of replaying one scenario N times under identical inputs."""

    scenario_id: str
    n_runs: int
    n_steps: int
    state_hashes_identical: bool
    first_state_divergence_step: int | None
    final_state_hash_identical: bool
    event_sequence_identical: bool
    fluxa_chain_root_identical: bool
    event_tree_root_identical: bool
    state_tree_root_identical: bool
    output_hash_identical: bool
    identity_hash_identical: bool
    qradle_output_hash_identical: bool
    qradle_chain_root_identical: bool
    distinct_identity_hashes: list[str]
    distinct_qradle_chain_roots: list[str]
    bundle_differences: dict[str, Any] = field(default_factory=dict)

    @property
    def deterministic_at_fluxa_level(self) -> bool:
        return (
            self.state_hashes_identical
            and self.final_state_hash_identical
            and self.event_sequence_identical
            and self.fluxa_chain_root_identical
            and self.event_tree_root_identical
            and self.state_tree_root_identical
            and self.output_hash_identical
            and self.identity_hash_identical
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["deterministic_at_fluxa_level"] = self.deterministic_at_fluxa_level
        return payload


def replay(
    loaded: LoadedSystem,
    scenario: Scenario,
    *,
    n_runs: int = 3,
    n_steps: int = 288,
    run_id: str = "determinism",
) -> ReplayReport:
    """Run ``scenario`` ``n_runs`` times with identical inputs and compare.

    The run *label* is held constant across replays, because an identical
    experiment is what determinism is a claim about; the separate
    label-independence property is covered by the test suite.
    """
    results = []
    for _ in range(n_runs):
        config = RunConfig(
            run_id=run_id,
            n_steps=n_steps,
            authorized=True,
            authorizer_id="determinism_experiment",
        )
        results.append(SimulationEngine(loaded, config).run(scenario))

    hashes = [r.state_hashes for r in results]
    divergence = None
    if len({tuple(h) for h in hashes}) > 1:
        for index, values in enumerate(zip(*hashes, strict=True)):
            if len(set(values)) > 1:
                divergence = index
                break

    sequences = [
        [(e.sequence, e.event_type.value, e.step, e.safety_level) for e in r.ledger]
        for r in results
    ]
    identities = sorted({r.provenance.identity_hash() for r in results})
    qradle_roots = sorted({r.provenance.qradle_chain_root for r in results})

    return ReplayReport(
        scenario_id=scenario.scenario_id,
        n_runs=n_runs,
        n_steps=n_steps,
        state_hashes_identical=len({tuple(h) for h in hashes}) == 1,
        first_state_divergence_step=divergence,
        final_state_hash_identical=len({r.provenance.final_state_hash for r in results}) == 1,
        event_sequence_identical=len({tuple(s) for s in sequences}) == 1,
        fluxa_chain_root_identical=len({r.provenance.event_chain_root for r in results}) == 1,
        event_tree_root_identical=len({r.provenance.event_tree_root for r in results}) == 1,
        state_tree_root_identical=len({r.provenance.state_tree_root for r in results}) == 1,
        output_hash_identical=len({r.provenance.output_hash for r in results}) == 1,
        identity_hash_identical=len(identities) == 1,
        qradle_output_hash_identical=len({r.provenance.qradle_output_hash for r in results}) == 1,
        qradle_chain_root_identical=len(qradle_roots) == 1,
        distinct_identity_hashes=identities,
        distinct_qradle_chain_roots=qradle_roots,
        bundle_differences=(
            diff_bundles(results[0].provenance, results[1].provenance) if n_runs > 1 else {}
        ),
    )


@dataclass
class ProvenanceExperimentReport:
    """The three required provenance tests, measured."""

    scenario_id: str
    n_events: int
    n_states: int

    # Test 1 -- untampered
    untampered_ledger_verifies: bool
    untampered_all_inclusion_proofs_verify: bool
    event_tree_root: str
    state_tree_root: str
    final_state_hash: str

    # Test 2 -- modified historical event
    tampered_event_index: int
    tampered_ledger_verifies: bool
    tamper_detection_messages: list[str]
    tamper_changes_chain_head: bool
    rehashed_tamper_detected: bool
    rehashed_tamper_messages: list[str]

    # Test 3 -- modified input parameter
    input_change_description: str
    identity_hash_before: str
    identity_hash_after: str
    identity_changed: bool
    changed_bundle_fields: list[str]

    # Substrate limitation
    qradle_proof_accepts_forged_path: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def provenance_experiment(
    loaded: LoadedSystem,
    scenario: Scenario,
    *,
    n_steps: int = 288,
) -> ProvenanceExperimentReport:
    """Execute the untampered / event-tamper / input-tamper provenance tests."""
    import copy
    import dataclasses
    import json

    from fluxa.config import DEFAULT_SYSTEM_CONFIG, load_system
    from qradle.core.merkle import MerkleChain, MerkleNode, MerkleProof

    def build(cfg_loaded: LoadedSystem, run_id: str):
        config = RunConfig(
            run_id=run_id, n_steps=n_steps, authorized=True, authorizer_id="provenance_experiment"
        )
        return SimulationEngine(cfg_loaded, config).run(scenario)

    clean = build(loaded, "prov-clean")

    # ---- Test 1: untampered -------------------------------------------
    ledger_ok, _ = clean.ledger.verify()
    tree = build_merkle_tree(clean.ledger.event_hashes())
    all_proofs_ok = all(
        verify_inclusion_proof(build_inclusion_proof(tree, i, h), tree.root)
        for i, h in enumerate(clean.ledger.event_hashes())
    )

    # ---- Test 2: modify one historical event ---------------------------
    tampered = build(loaded, "prov-tampered")
    head_before = tampered.ledger.root_hash()
    victim = len(tampered.ledger) // 2
    forged = copy.deepcopy(tampered.ledger.events[victim].to_dict())
    forged["payload"] = {**forged.get("payload", {}), "unserved_mw": 0.0}
    tampered.ledger._force_replace_node_data(victim, forged)
    tampered_ok, tamper_messages = tampered.ledger.verify()

    # A more capable attacker rehashes the node they edited.
    rehashed = build(loaded, "prov-rehashed")
    nodes = list(rehashed.ledger._nodes)
    target = 5
    nodes[target] = MerkleNode(
        data={**nodes[target].data, "payload": {"tampered": True}},
        timestamp=nodes[target].timestamp,
        previous_hash=nodes[target].previous_hash,
    )
    rehashed.ledger._nodes = nodes
    rehashed_ok, rehashed_messages = rehashed.ledger.verify()

    # ---- Test 3: modify one input parameter ----------------------------
    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    original_capacity = doc["batteries"][0]["energy_capacity_mwh"]
    doc["batteries"][0]["energy_capacity_mwh"] = original_capacity + 1.0
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        variant_path = Path(tmp) / "variant.json"
        variant_path.write_text(json.dumps(doc))
        variant = build(load_system(variant_path), "prov-input-changed")

    # ---- Substrate limitation: QRADLE proof does not recompute ----------
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
    qradle_accepts_forgery = bool(forged_proof.verify(chain.get_root_hash()))

    return ProvenanceExperimentReport(
        scenario_id=scenario.scenario_id,
        n_events=len(clean.ledger),
        n_states=len(clean.states),
        untampered_ledger_verifies=ledger_ok,
        untampered_all_inclusion_proofs_verify=all_proofs_ok,
        event_tree_root=clean.provenance.event_tree_root,
        state_tree_root=clean.provenance.state_tree_root,
        final_state_hash=clean.provenance.final_state_hash,
        tampered_event_index=victim,
        tampered_ledger_verifies=tampered_ok,
        tamper_detection_messages=tamper_messages,
        tamper_changes_chain_head=(tampered.ledger.root_hash() != head_before),
        rehashed_tamper_detected=not rehashed_ok,
        rehashed_tamper_messages=rehashed_messages[:4],
        input_change_description=(
            f"BESS_1.energy_capacity_mwh {original_capacity} -> {original_capacity + 1.0}"
        ),
        identity_hash_before=clean.provenance.identity_hash(),
        identity_hash_after=variant.provenance.identity_hash(),
        identity_changed=(
            variant.provenance.identity_hash() != clean.provenance.identity_hash()
        ),
        changed_bundle_fields=sorted(diff_bundles(clean.provenance, variant.provenance)),
        qradle_proof_accepts_forged_path=qradle_accepts_forgery,
    )

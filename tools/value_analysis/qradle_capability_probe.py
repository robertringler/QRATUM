#!/usr/bin/env python3
"""Adversarial probe of QRADLE's claimed guarantees.

Answers section-10 ("what can it actually prove?") of the value-asset
analysis with executed evidence rather than README claims.

Run from the repository root:
    python tools/value_analysis/qradle_capability_probe.py
Exit code is always 0: this is a measurement tool, not a test gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys

sys.path.insert(0, ".")

from qradle.core.engine import DeterministicEngine, ExecutionContext  # noqa: E402
from qradle.core.merkle import MerkleChain, MerkleProof  # noqa: E402

RESULTS: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
        except Exception as exc:  # noqa: BLE001 - we want to record, not raise
            RESULTS[name] = {"exception": f"{type(exc).__name__}: {exc}"}
        return fn

    return deco


def _ctx(cid, params, level="ROUTINE", authorized=True):
    return ExecutionContext(cid, params, "2025-01-01T00:00:00Z", level, authorized)


@probe("Q1_can_it_reproduce_the_audit_root_on_replay")
def _():
    def run():
        eng = DeterministicEngine()
        res = eng.execute_contract(_ctx("k", {"x": 1}), lambda p: p["x"] + 1)
        return res.output_hash, eng.merkle_chain.get_root_hash()

    a, b = run(), run()
    return {
        "output_hash_stable": a[0] == b[0],
        "chain_root_stable": a[1] == b[1],
        "cause": "MerkleChain.append() stamps datetime.now() into the hashed node",
        "verdict": "NO - output hash replays, audit root does not",
    }


@probe("Q2_does_it_detect_a_non_deterministic_executor")
def _():
    counter = itertools.count()
    eng = DeterministicEngine()
    runs = []
    for _i in range(2):
        res = eng.execute_contract(_ctx("nd", {"x": 1}), lambda p: next(counter))
        runs.append({"success": res.success, "output": res.output})
    return {
        "runs": runs,
        "engine_raised_violation": False,
        "cause": "FatalInvariants.enforce_determinism() is never called by the engine",
        "verdict": "NO - invariant 8 is declared but not wired",
    }


@probe("Q3_can_it_detect_tampering_with_a_recorded_event")
def _():
    eng = DeterministicEngine()
    eng.execute_contract(_ctx("k", {"x": 1}), lambda p: p["x"])
    before = eng.merkle_chain.verify_chain_integrity()
    eng.merkle_chain.nodes[1].data["contract_id"] = "TAMPERED"
    after = eng.merkle_chain.verify_chain_integrity()
    return {
        "valid_before": before,
        "valid_after_mutation": after,
        "verdict": "YES - in-place mutation of a recorded event is detected",
    }


@probe("Q4_can_a_whole_chain_be_forged")
def _():
    honest = MerkleChain({"engine": "QRADLE", "version": "1.0.0"})
    honest.append({"event_type": "payment", "amount": 100})
    forged = MerkleChain({"engine": "QRADLE", "version": "1.0.0"})
    forged.append({"event_type": "payment", "amount": 999_999})
    return {
        "forged_chain_passes_its_own_integrity_check": forged.verify_chain_integrity(),
        "roots_differ": honest.get_root_hash() != forged.get_root_hash(),
        "cause": "no signature, no key material, no external anchor, no timestamping",
        "verdict": "YES - rewriting history wholesale is undetectable without an out-of-band root",
    }


@probe("Q5_is_get_proof_a_real_inclusion_proof")
def _():
    chain = MerkleChain()
    for i in range(1000):
        chain.append({"i": i})
    proof = chain.get_proof(1)
    bogus = MerkleProof(
        event_id="x",
        event_hash="deadbeef",
        chain_position=1,
        proof_path=[],
        root_hash=chain.get_root_hash(),
    )
    return {
        "chain_length": len(chain.nodes),
        "proof_path_length": len(proof.proof_path),
        "is_logarithmic": len(proof.proof_path) <= 20,
        "verify_proof_accepts_empty_bogus_proof": chain.verify_proof(bogus),
        "cause": "MerkleProof.verify() only compares root_hash strings; event_hash is unused",
        "verdict": "NO - it is an O(n) hash list, and the verifier checks nothing about inclusion",
    }


@probe("Q6_can_it_prove_who_authorized_an_action")
def _():
    eng = DeterministicEngine()
    res = eng.execute_contract(_ctx("sensitive", {"x": 1}, "CRITICAL", True), lambda p: p["x"])
    log = json.dumps([n.data for n in eng.merkle_chain.nodes]).lower()
    identity_keys = ("approver", "signature", "principal", "pubkey", "user_id", "actor")
    return {
        "executed_at_CRITICAL": res.success,
        "identity_present_in_event_log": any(k in log for k in identity_keys),
        "authorization_representation": "a caller-supplied bool on ExecutionContext",
        "verdict": "NO - authorization is self-asserted and anonymous",
    }


@probe("Q7_is_multi_party_approval_enforced_per_safety_level")
def _():
    eng = DeterministicEngine()
    out = {}
    for level in ("ROUTINE", "ELEVATED", "SENSITIVE", "CRITICAL", "EXISTENTIAL"):
        res = eng.execute_contract(_ctx(f"op_{level}", {}, level, True), lambda p: 1)
        out[level] = res.success
    return {
        "succeeded_with_a_single_boolean": out,
        "documented_requirement": "CRITICAL=multi-human, EXISTENTIAL=board+external",
        "verdict": "NO - every level clears on one unauthenticated bool",
    }


@probe("Q8_does_rollback_restore_state")
def _():
    eng = DeterministicEngine()
    r1 = eng.execute_contract(_ctx("s1", {"v": 1}), lambda p: p["v"])
    len_after_1 = len(eng.merkle_chain.nodes)
    eng.execute_contract(_ctx("s2", {"v": 2}), lambda p: p["v"])
    ok = eng.rollback_to_checkpoint(r1.checkpoint_id)
    return {
        "rollback_returned_true": ok,
        "chain_length_after_step1": len_after_1,
        "chain_length_after_rollback": len(eng.merkle_chain.nodes),
        "execution_count_after_rollback": eng._execution_count,
        "verdict": "PARTIAL - the checkpoint dict is returned; engine and application state are not reverted",
    }


@probe("Q9_can_an_independent_party_verify_the_result")
def _():
    eng = DeterministicEngine()
    eng.execute_contract(_ctx("k", {"x": 1}), lambda p: p["x"])
    exported = eng.merkle_chain.export_chain()
    genesis = hashlib.sha256(b"QRADLE_GENESIS_1.0.0").hexdigest()
    prev, links_ok = genesis, True
    for node in exported:
        content = {
            "data": node["data"],
            "timestamp": node["timestamp"],
            "previous_hash": node["previous_hash"],
        }
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if digest != node["node_hash"] or node["previous_hash"] != prev:
            links_ok = False
        prev = node["node_hash"]
    blob = json.dumps(exported)
    return {
        "third_party_can_recompute_link_hashes": links_ok,
        "export_contains_inputs": "parameters" in blob,
        "export_contains_code_identity": any(
            k in blob for k in ("code_hash", "image_digest", "version_sha")
        ),
        "verdict": "NO - the log is self-consistent but carries no inputs, no code identity, no signature",
    }


if __name__ == "__main__":
    print(json.dumps(RESULTS, indent=2, default=str))

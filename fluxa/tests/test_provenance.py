"""Provenance: Merkle structures, inclusion proofs, and tamper evidence.

These tests implement the three provenance experiments the study requires:
  Test 1 -- an untampered simulation verifies.
  Test 2 -- modifying one historical event makes verification fail.
  Test 3 -- modifying one input parameter changes the run's identity.

They also record a measured weakness in QRADLE's own proof object, which
compares stored roots rather than recomputing one.
"""

from __future__ import annotations

import copy

import pytest

from fluxa.provenance import (
    EMPTY_TREE_ROOT,
    InclusionProof,
    build_inclusion_proof,
    build_merkle_tree,
    diff_bundles,
    environment_fingerprint,
    leaf_hash,
    node_hash,
    sha256_of,
    verify_inclusion_proof,
)
from fluxa.tests.conftest import make_engine


# --------------------------------------------------------------- tree shape
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 17, 100])
def test_tree_builds_and_every_leaf_proof_verifies(n):
    leaves = [f"{i:064x}" for i in range(n)]
    tree = build_merkle_tree(leaves)
    assert tree.n_leaves == n
    assert tree.levels[-1] and len(tree.levels[-1]) == 1
    for i, payload in enumerate(leaves):
        proof = build_inclusion_proof(tree, i, payload)
        assert verify_inclusion_proof(proof, tree.root), f"leaf {i} of {n} failed"


def test_empty_tree_has_a_distinct_root():
    assert build_merkle_tree([]).root == EMPTY_TREE_ROOT


def test_leaf_and_node_hashes_are_domain_separated():
    """Without domain separation an internal node could be presented as a
    leaf. The tags must make the two preimage spaces disjoint."""
    value = f"{7:064x}"
    assert leaf_hash(value) != node_hash(value, value)


def test_changing_any_leaf_changes_the_root():
    leaves = [f"{i:064x}" for i in range(9)]
    base = build_merkle_tree(leaves).root
    for i in range(9):
        mutated = list(leaves)
        mutated[i] = f"{999 + i:064x}"
        assert build_merkle_tree(mutated).root != base, f"leaf {i} change not detected"


def test_reordering_leaves_changes_the_root():
    leaves = [f"{i:064x}" for i in range(6)]
    swapped = [leaves[1], leaves[0], *leaves[2:]]
    assert build_merkle_tree(swapped).root != build_merkle_tree(leaves).root


def test_proof_with_a_substituted_leaf_fails():
    leaves = [f"{i:064x}" for i in range(11)]
    tree = build_merkle_tree(leaves)
    honest = build_inclusion_proof(tree, 4, leaves[4])
    forged = InclusionProof(
        index=4,
        leaf_payload=f"{4242:064x}",
        path=honest.path,
        path_is_right=honest.path_is_right,
        root=tree.root,
    )
    assert not verify_inclusion_proof(forged, tree.root)


def test_proof_with_a_corrupted_sibling_fails():
    leaves = [f"{i:064x}" for i in range(11)]
    tree = build_merkle_tree(leaves)
    honest = build_inclusion_proof(tree, 4, leaves[4])
    corrupted = InclusionProof(
        index=4,
        leaf_payload=honest.leaf_payload,
        path=(f"{0:064x}", *honest.path[1:]),
        path_is_right=honest.path_is_right,
        root=tree.root,
    )
    assert not verify_inclusion_proof(corrupted, tree.root)


def test_proof_against_a_foreign_root_fails():
    leaves = [f"{i:064x}" for i in range(8)]
    tree = build_merkle_tree(leaves)
    other = build_merkle_tree([f"{i + 1000:064x}" for i in range(8)])
    proof = build_inclusion_proof(tree, 2, leaves[2])
    assert not verify_inclusion_proof(proof, other.root)


def test_out_of_range_leaf_index_is_rejected():
    tree = build_merkle_tree([f"{i:064x}" for i in range(4)])
    with pytest.raises(IndexError):
        build_inclusion_proof(tree, 4, f"{4:064x}")


def test_qradle_proof_object_does_not_recompute_a_root():
    """Measured limitation of the substrate, recorded rather than assumed.

    ``qradle.core.merkle.MerkleProof.verify`` returns
    ``self.root_hash == claimed_root``. It therefore accepts any proof whose
    stored root matches, regardless of whether its ``event_hash`` or
    ``proof_path`` are consistent with that root. This is why FLUXA carries
    its own recomputing inclusion proof.
    """
    from qradle.core.merkle import MerkleChain, MerkleProof

    chain = MerkleChain({"t": "genesis"})
    chain.append({"event": "a"})
    chain.append({"event": "b"})
    genuine = chain.get_proof(1)

    forged = MerkleProof(
        event_id="fabricated",
        event_hash="00" * 32,
        chain_position=1,
        proof_path=["ff" * 32],
        root_hash=genuine.root_hash,
    )
    assert forged.verify(chain.get_root_hash()) is True, (
        "QRADLE's proof now recomputes a root; this limitation and the report "
        "section that cites it must be updated"
    )
    assert chain.verify_proof(forged) is True


# ------------------------------------------------- experiment 1: untampered
def test_untampered_simulation_verifies(loaded, scenarios):
    result = make_engine(loaded, run_id="prov-clean", n_steps=48).run(scenarios["A_BASELINE"])
    ok, problems = result.ledger.verify()
    assert ok and problems == []

    tree = build_merkle_tree(result.ledger.event_hashes())
    assert tree.root == result.provenance.event_tree_root
    for i, h in enumerate(result.ledger.event_hashes()):
        assert verify_inclusion_proof(build_inclusion_proof(tree, i, h), tree.root)

    states = build_merkle_tree(result.state_hashes)
    assert states.root == result.provenance.state_tree_root
    assert result.states[-1].state_hash() == result.provenance.final_state_hash


# ------------------------------------------------- experiment 2: event tamper
def test_modifying_one_historical_event_fails_verification(loaded, scenarios):
    result = make_engine(loaded, run_id="prov-tamper", n_steps=48).run(scenarios["A_BASELINE"])
    assert result.ledger.verify()[0]

    victim = len(result.ledger) // 2
    original_root = result.ledger.root_hash()
    original_tree_root = build_merkle_tree(result.ledger.event_hashes()).root

    forged = copy.deepcopy(result.ledger.events[victim].to_dict())
    forged["payload"] = {**forged.get("payload", {}), "total_load_mw": 1.0}
    result.ledger._force_replace_node_data(victim, forged)

    ok, problems = result.ledger.verify()
    assert not ok, "tampering with a historical event was not detected"
    assert any(f"node {victim}" in p for p in problems)
    # The chain head is unchanged -- a head comparison alone would miss this,
    # which is precisely why per-node content verification is required.
    assert result.ledger.root_hash() == original_root
    # A recomputed tree over the per-node hashes also still matches, because
    # the forged node's stored hash was not updated. The detection signal is
    # the per-node content check, and the test asserts which one fires.
    assert build_merkle_tree(result.ledger.event_hashes()).root == original_tree_root
    assert problems == [f"node {victim}: content hash does not match data"]


def test_rehashing_a_tampered_event_breaks_the_chain_linkage(loaded, scenarios):
    """A more capable attacker rehashes the node they edited. The chain's
    previous_hash linkage must then expose the break at the next node."""
    from qradle.core.merkle import MerkleNode

    result = make_engine(loaded, run_id="prov-rehash", n_steps=48).run(scenarios["A_BASELINE"])
    victim = 5
    nodes = list(result.ledger._nodes)
    forged_data = {**nodes[victim].data, "payload": {"tampered": True}}
    nodes[victim] = MerkleNode(
        data=forged_data,
        timestamp=nodes[victim].timestamp,
        previous_hash=nodes[victim].previous_hash,
    )
    result.ledger._nodes = nodes

    ok, problems = result.ledger.verify()
    assert not ok
    assert any(f"node {victim + 1}" in p and "previous_hash" in p for p in problems)


def test_truncating_the_ledger_changes_the_tree_root(loaded, scenarios):
    result = make_engine(loaded, run_id="prov-truncate", n_steps=48).run(
        scenarios["A_BASELINE"]
    )
    full = build_merkle_tree(result.ledger.event_hashes()).root
    truncated = build_merkle_tree(result.ledger.event_hashes()[:-1]).root
    assert full != truncated


def test_tampering_with_one_state_changes_the_state_tree_root(loaded, scenarios):
    import dataclasses

    result = make_engine(loaded, run_id="prov-state", n_steps=48).run(scenarios["A_BASELINE"])
    hashes = list(result.state_hashes)
    assert build_merkle_tree(hashes).root == result.provenance.state_tree_root

    nudged = dataclasses.replace(result.states[10], total_load_mw=result.states[10].total_load_mw + 1e-9)
    hashes[10] = nudged.state_hash()
    assert build_merkle_tree(hashes).root != result.provenance.state_tree_root


# ------------------------------------------------- experiment 3: input tamper
@pytest.mark.parametrize(
    "kwargs",
    [
        {"wind_seed": 424_242},
        {"horizon_steps": 6},
        {"timestep_s": 900.0, "n_steps": 16},
        {"stochastic_profiles": False},
    ],
)
def test_changing_one_input_changes_the_run_identity(loaded, scenarios, kwargs):
    base = make_engine(loaded, run_id="prov-base", n_steps=48).run(scenarios["A_BASELINE"])
    steps = kwargs.pop("n_steps", 48)
    changed = make_engine(loaded, run_id="prov-changed", n_steps=steps, **kwargs).run(
        scenarios["A_BASELINE"]
    )
    assert changed.provenance.identity_hash() != base.provenance.identity_hash()
    assert diff_bundles(base.provenance, changed.provenance)


def test_changing_the_system_configuration_changes_the_identity(loaded, scenarios, tmp_path):
    import json

    from fluxa.config import DEFAULT_SYSTEM_CONFIG, load_system

    base = make_engine(loaded, run_id="cfg-base", n_steps=48).run(scenarios["A_BASELINE"])

    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    doc["generators"][0]["marginal_cost_usd_per_mwh"] = 49.0
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(doc))
    variant = load_system(path)

    assert variant.config_hash != loaded.config_hash
    assert variant.model_hash != loaded.model_hash

    changed = make_engine(variant, run_id="cfg-changed", n_steps=48).run(
        scenarios["A_BASELINE"]
    )
    assert changed.provenance.identity_hash() != base.provenance.identity_hash()
    fields = diff_bundles(base.provenance, changed.provenance)
    assert "config_hash" in fields and "model_hash" in fields


def test_changing_the_scenario_changes_the_identity(loaded, scenarios):
    a = make_engine(loaded, run_id="sc-a", n_steps=48).run(scenarios["A_BASELINE"])
    b = make_engine(loaded, run_id="sc-b", n_steps=48).run(scenarios["B_SOLAR_SHOCK"])
    assert a.provenance.identity_hash() != b.provenance.identity_hash()
    assert "scenario_hash" in diff_bundles(a.provenance, b.provenance)


# ------------------------------------------------------------------ bundle
def test_bundle_records_the_numerical_environment(loaded, scenarios):
    result = make_engine(loaded, run_id="prov-env", n_steps=24).run(scenarios["A_BASELINE"])
    env = result.provenance.environment
    assert env == environment_fingerprint()
    for key in ("python_version", "numpy_version", "scipy_version", "platform"):
        assert env[key]


def test_bundle_records_the_solver_identity(loaded, scenarios):
    result = make_engine(loaded, run_id="prov-solver", n_steps=24).run(scenarios["A_BASELINE"])
    solver = result.provenance.solver
    assert solver["method"] == "highs"
    assert solver["library"] == "scipy.optimize.linprog"


def test_identity_hash_excludes_the_environment_so_it_is_diagnosable(loaded, scenarios):
    """An environment change must be visible as a *named* difference, not
    merely as a different identity, so "the model changed" and "SciPy
    changed" can be told apart."""
    result = make_engine(loaded, run_id="prov-id", n_steps=24).run(scenarios["A_BASELINE"])
    import dataclasses

    other_env = dataclasses.replace(
        result.provenance, environment={**result.provenance.environment, "scipy_version": "0.0.0"}
    )
    assert other_env.identity_hash() == result.provenance.identity_hash()
    assert other_env.bundle_hash() != result.provenance.bundle_hash()
    assert "environment" in diff_bundles(result.provenance, other_env)


def test_canonical_hash_is_insensitive_to_key_order():
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})


def test_canonical_hash_is_sensitive_to_the_last_float_bit():
    assert sha256_of({"x": 1.0}) != sha256_of({"x": 1.0 + 2**-52})

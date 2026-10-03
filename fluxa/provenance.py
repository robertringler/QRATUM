"""Cryptographic provenance for a FLUXA simulation run.

Two structures are built here.

**1. A binary Merkle tree over the event hashes.**
QRADLE's ``MerkleProof.verify`` compares a stored ``root_hash`` field against
a claimed root and returns the comparison. It does not recompute anything
from the proof path, so a proof object that carries a stale root verifies
against that stale root and a modified event is not detected by the proof
alone. :func:`build_merkle_tree` and :func:`verify_inclusion_proof` here
recompute the root from the leaf and the sibling path, which is what makes an
inclusion proof load-bearing. Domain separation tags distinguish leaf hashes
from internal-node hashes, closing the standard second-preimage attack on
Merkle trees with duplicated odd leaves.

**2. A provenance bundle.**
A single document binding together everything needed to decide whether two
runs are the same run: the hashes of the inputs, the model, the scenario, the
software environment, the event sequence, every timestep state, and the final
state. The bundle's own ``bundle_hash`` is the run's cryptographic identity.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy
import scipy

#: Domain-separation prefixes. Hashing a leaf and hashing an internal node
#: with the same function would let an attacker present an internal node as a
#: leaf; the tags make the two preimage spaces disjoint.
_LEAF_TAG = b"\x00FLUXA_LEAF"
_NODE_TAG = b"\x01FLUXA_NODE"

#: Hash of an empty tree.
EMPTY_TREE_ROOT = hashlib.sha256(b"\x02FLUXA_EMPTY").hexdigest()


def leaf_hash(payload: str) -> str:
    """Tagged hash of a leaf's hex-encoded content hash."""
    return hashlib.sha256(_LEAF_TAG + payload.encode("utf-8")).hexdigest()


def node_hash(left: str, right: str) -> str:
    """Tagged hash of two child hashes."""
    return hashlib.sha256(_NODE_TAG + left.encode("utf-8") + right.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MerkleTree:
    """A complete binary Merkle tree over an ordered list of leaves.

    Attributes:
        levels: ``levels[0]`` are the tagged leaf hashes; each subsequent
            level is the parent level. ``levels[-1]`` has exactly one element,
            the root. An odd level promotes its last element by pairing it
            with itself.
        n_leaves: Number of original leaves.
    """

    levels: tuple[tuple[str, ...], ...]
    n_leaves: int

    @property
    def root(self) -> str:
        if self.n_leaves == 0:
            return EMPTY_TREE_ROOT
        return self.levels[-1][0]


def build_merkle_tree(leaf_payloads: list[str]) -> MerkleTree:
    """Build a Merkle tree over ``leaf_payloads`` (hex content hashes)."""
    if not leaf_payloads:
        return MerkleTree(levels=((),), n_leaves=0)
    level = tuple(leaf_hash(p) for p in leaf_payloads)
    levels: list[tuple[str, ...]] = [level]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(node_hash(left, right))
        level = tuple(nxt)
        levels.append(level)
    return MerkleTree(levels=tuple(levels), n_leaves=len(leaf_payloads))


@dataclass(frozen=True)
class InclusionProof:
    """Proof that a leaf is included in a tree with a given root.

    Attributes:
        index: Leaf position.
        leaf_payload: The leaf's content hash (pre-tag).
        path: Sibling hashes bottom-up.
        path_is_right: For each path element, True if the sibling is the right
            child (so the running hash is the left child).
        root: The root the proof was generated against.
    """

    index: int
    leaf_payload: str
    path: tuple[str, ...]
    path_is_right: tuple[bool, ...]
    root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "leaf_payload": self.leaf_payload,
            "path": list(self.path),
            "path_is_right": list(self.path_is_right),
            "root": self.root,
        }


def build_inclusion_proof(tree: MerkleTree, index: int, leaf_payload: str) -> InclusionProof:
    """Generate an inclusion proof for ``index``.

    Raises:
        IndexError: if ``index`` is outside the tree.
    """
    if not 0 <= index < tree.n_leaves:
        raise IndexError(f"leaf index {index} outside tree of {tree.n_leaves} leaves")
    path: list[str] = []
    is_right: list[bool] = []
    position = index
    for level in tree.levels[:-1]:
        sibling_index = position + 1 if position % 2 == 0 else position - 1
        # An odd-sized level pairs its final element with itself.
        sibling = level[sibling_index] if sibling_index < len(level) else level[position]
        path.append(sibling)
        is_right.append(position % 2 == 0)
        position //= 2
    return InclusionProof(
        index=index,
        leaf_payload=leaf_payload,
        path=tuple(path),
        path_is_right=tuple(is_right),
        root=tree.root,
    )


def verify_inclusion_proof(proof: InclusionProof, claimed_root: str) -> bool:
    """Recompute the root from the leaf and the sibling path.

    Unlike a root-equality check, this fails if the leaf content changed, if a
    sibling changed, or if the path was reordered.
    """
    running = leaf_hash(proof.leaf_payload)
    for sibling, running_is_left in zip(proof.path, proof.path_is_right, strict=True):
        running = (
            node_hash(running, sibling) if running_is_left else node_hash(sibling, running)
        )
    return running == claimed_root


def environment_fingerprint() -> dict[str, Any]:
    """Software environment recorded with every run.

    A provenance chain that does not pin the numerical stack cannot
    distinguish "the model changed" from "SciPy changed".
    """
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
    }


def canonical_json(payload: Any) -> str:
    """Canonical JSON encoding: sorted keys, tight separators, exact floats."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_of(payload: Any) -> str:
    """SHA-256 over :func:`canonical_json` of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProvenanceBundle:
    """The cryptographic identity of one simulation run.

    Attributes:
        run_id: Caller-supplied identifier. Deliberately *excluded* from
            :meth:`identity_hash` so two runs of the same experiment under
            different run ids can be compared.
        config_hash: Hash of the system configuration document as loaded.
        model_hash: Hash of the validated physical/economic model.
        scenario_hash: Hash of the scenario definition.
        profile_hash: Hash of the exogenous driver series actually used.
        run_parameters: Horizon, timestep, seeds, solver settings.
        environment: :func:`environment_fingerprint` output.
        solver: Solver identity from the dispatch engine.
        event_chain_root: Head hash of the FLUXA event ledger.
        event_tree_root: Binary Merkle-tree root over the event hashes.
        state_tree_root: Binary Merkle-tree root over per-timestep state hashes.
        final_state_hash: Hash of the last timestep's state.
        output_hash: Hash of the aggregated run metrics.
        n_events: Event count.
        n_states: Timestep count.
        qradle_output_hash: ``ExecutionResult.output_hash`` from QRADLE.
        qradle_chain_root: Root of QRADLE's own (wall-clock-stamped) chain.
    """

    run_id: str
    config_hash: str
    model_hash: str
    scenario_hash: str
    profile_hash: str
    run_parameters: dict[str, Any]
    environment: dict[str, Any]
    solver: dict[str, Any]
    event_chain_root: str
    event_tree_root: str
    state_tree_root: str
    final_state_hash: str
    output_hash: str
    n_events: int
    n_states: int
    qradle_output_hash: str = ""
    qradle_chain_root: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    #: Fields that participate in :meth:`identity_hash`. Excluded:
    #: ``run_id`` (a label), ``environment`` (compared separately so an
    #: environment change is diagnosable rather than just "different"), and
    #: ``qradle_chain_root`` (wall-clock dependent by construction).
    IDENTITY_FIELDS = (
        "config_hash",
        "model_hash",
        "scenario_hash",
        "profile_hash",
        "run_parameters",
        "solver",
        "event_chain_root",
        "event_tree_root",
        "state_tree_root",
        "final_state_hash",
        "output_hash",
        "n_events",
        "n_states",
        "qradle_output_hash",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "model_hash": self.model_hash,
            "scenario_hash": self.scenario_hash,
            "profile_hash": self.profile_hash,
            "run_parameters": self.run_parameters,
            "environment": self.environment,
            "solver": self.solver,
            "event_chain_root": self.event_chain_root,
            "event_tree_root": self.event_tree_root,
            "state_tree_root": self.state_tree_root,
            "final_state_hash": self.final_state_hash,
            "output_hash": self.output_hash,
            "n_events": self.n_events,
            "n_states": self.n_states,
            "qradle_output_hash": self.qradle_output_hash,
            "qradle_chain_root": self.qradle_chain_root,
            "extra": self.extra,
            "identity_hash": self.identity_hash(),
        }

    def identity_hash(self) -> str:
        """Hash over :attr:`IDENTITY_FIELDS`: the run's reproducible identity."""
        subset = {name: getattr(self, name) for name in self.IDENTITY_FIELDS}
        return sha256_of(subset)

    def bundle_hash(self) -> str:
        """Hash over the entire bundle, environment and run id included."""
        payload = self.to_dict()
        payload.pop("identity_hash", None)
        return sha256_of(payload)


def diff_bundles(a: ProvenanceBundle, b: ProvenanceBundle) -> dict[str, tuple[Any, Any]]:
    """Field-by-field difference between two bundles.

    Returns only the fields that differ, so a determinism report can name the
    cause rather than asserting inequality.
    """
    out: dict[str, tuple[Any, Any]] = {}
    left, right = a.to_dict(), b.to_dict()
    for key in sorted(set(left) | set(right)):
        if key == "run_id":
            continue
        if left.get(key) != right.get(key):
            out[key] = (left.get(key), right.get(key))
    return out

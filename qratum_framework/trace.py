"""trace.py — unified TraceEntry schema and Merkle-chained ledger.

The plan calls for **one TraceEntry schema** with a stable layout
(step, intent, action, pre/post state hash, Φ verdict, failure_mode,
metrics, signatures) and **one append-only Merkle-chained provenance log**
shared by every subsystem.  This module provides exactly that.

Key properties (deterministic by construction):

* Hashing uses canonical JSON (`sort_keys=True`, no whitespace) so two runs
  with identical inputs produce *byte-identical* hash chains.
* The chain is bound by hashing each entry together with the previous
  entry's hash; tampering with any entry invalidates every subsequent one.
* No timestamp is included in the canonical hash payload — determinism is
  preferred over wall-clock provenance.  Wall-clock metadata, if needed,
  travels in a separate ``metadata`` field that is excluded from hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

#: Canonical genesis hash for the head of an empty Merkle ledger.  The empty
#: string SHA-256 is used so that the genesis is independent of any state.
GENESIS_HASH: str = hashlib.sha256(b"").hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic JSON encoding suitable for hashing.

    Uses ``sort_keys=True`` and the most compact separators so the encoding
    is independent of Python dict iteration order or platform whitespace
    conventions.  Non-JSON-serialisable values are stringified through
    ``default=str``; this is intentional — every legitimate field of a
    UnifiedTraceEntry is JSON-native already, but defensive coercion makes
    the hash robust to backend payload variation.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class UnifiedTraceEntry:
    """Stable, schema-versioned trace entry.

    Field semantics:

    * ``step`` — monotonic non-negative tick index within a run.
    * ``intent_raw`` — the original intent string handed to the pipeline.
    * ``intent`` — the parsed intent symbol (or ``None`` on parse failure).
    * ``action`` — the action emitted by the controller (string form).
    * ``status`` — one of ``OK | TYPE_I | TYPE_II | TYPE_III | TYPE_IV``.
    * ``failure_mode`` — symbolic failure name when ``status != OK``.
    * ``pre_state_hash`` / ``post_state_hash`` — canonical-JSON SHA-256 of the
      backend-provided pre- and post-state dicts.  Independent of, and
      complementary to, the chain hash.
    * ``phi_verdict`` — boolean Φ verdict (``True`` if all constraints
      satisfied at the post-state, ``False`` otherwise).
    * ``violated_constraints`` — names of constraints that fired.
    * ``metrics`` — backend-supplied numeric metrics (loss, drift, SF, etc).
    * ``signatures`` — slot for biokey/FIDO2 signatures; carried opaquely.
    * ``metadata`` — free-form metadata *excluded from canonical hashing*
      (wall-clock, host, run-id, etc).  Tampering with metadata does not
      break the chain — only the canonical payload does.
    """

    step: int
    intent_raw: str
    intent: Optional[str]
    action: str
    status: str
    failure_mode: Optional[str]
    pre_state_hash: str
    post_state_hash: Optional[str]
    phi_verdict: bool
    violated_constraints: Tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    signatures: Tuple[str, ...] = ()
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # --- hashing ---------------------------------------------------------

    def canonical_payload(self) -> Dict[str, Any]:
        """Return the canonical, hashable representation of this entry.

        The ``metadata`` field is intentionally omitted so wall-clock or
        host-specific data cannot perturb the Merkle chain.
        """
        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "intent_raw": self.intent_raw,
            "intent": self.intent,
            "action": self.action,
            "status": self.status,
            "failure_mode": self.failure_mode,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "phi_verdict": bool(self.phi_verdict),
            "violated_constraints": list(self.violated_constraints),
            "metrics": {k: float(v) for k, v in dict(self.metrics).items()},
            "signatures": list(self.signatures),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict view suitable for JSON export (incl. metadata)."""
        d = self.canonical_payload()
        d["metadata"] = dict(self.metadata)
        return d


def hash_state(state: Any) -> str:
    """Canonical SHA-256 hash of a backend state.

    Accepts dicts, dataclass instances (any with ``__dict__`` or
    ``_asdict``), or anything ``str``-coercible.  ``None`` hashes to the
    empty-string digest so that a missing post-state is still deterministic.
    """
    if state is None:
        return GENESIS_HASH
    if hasattr(state, "_asdict"):
        payload = state._asdict()
    elif hasattr(state, "__dict__"):
        payload = dict(state.__dict__)
    elif isinstance(state, Mapping):
        payload = dict(state)
    else:
        payload = {"repr": repr(state)}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def hash_entry(entry: UnifiedTraceEntry, prev_hash: str) -> str:
    """Compute the chain hash binding an entry to its predecessor.

    ``chain_hash_n = SHA256( prev_hash || canonical_payload(entry_n) )``

    Returns a hex digest.  The genesis predecessor is ``GENESIS_HASH``.
    """
    payload = _canonical_json({"prev": prev_hash, "entry": entry.canonical_payload()})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _LedgerNode:
    entry: UnifiedTraceEntry
    chain_hash: str
    prev_hash: str


class MerkleLedger:
    """Append-only Merkle-chained ledger of UnifiedTraceEntries.

    The ledger is *the* canonical provenance log.  Every subsystem in the
    operational framework appends through this object; nothing else writes
    to provenance.  Verification is O(n) and deterministic.
    """

    def __init__(self, genesis: str = GENESIS_HASH) -> None:
        self._genesis = genesis
        self._nodes: List[_LedgerNode] = []

    # --- mutation -------------------------------------------------------

    def append(self, entry: UnifiedTraceEntry) -> str:
        """Append ``entry`` and return its chain hash."""
        prev = self.head()
        chain_hash = hash_entry(entry, prev)
        self._nodes.append(_LedgerNode(entry=entry, chain_hash=chain_hash, prev_hash=prev))
        return chain_hash

    def extend(self, entries: Iterable[UnifiedTraceEntry]) -> None:
        for e in entries:
            self.append(e)

    # --- inspection -----------------------------------------------------

    def head(self) -> str:
        """Hash of the most recent entry, or the genesis if empty."""
        return self._nodes[-1].chain_hash if self._nodes else self._genesis

    def height(self) -> int:
        return len(self._nodes)

    @property
    def entries(self) -> List[UnifiedTraceEntry]:
        return [n.entry for n in self._nodes]

    def iter_with_hashes(self) -> Iterable[Tuple[UnifiedTraceEntry, str, str]]:
        """Yield ``(entry, prev_hash, chain_hash)`` triples in append order.

        Public, read-only audit cursor over the ledger.  External verifiers
        (the CLI's ``ledger`` subcommand, third-party reproducers, …) use
        this instead of poking at internal nodes so that the chain layout
        can evolve without breaking callers.
        """
        for node in self._nodes:
            yield node.entry, node.prev_hash, node.chain_hash

    def __len__(self) -> int:  # pragma: no cover — convenience
        return len(self._nodes)

    def __iter__(self):  # pragma: no cover — convenience
        return (n.entry for n in self._nodes)

    # --- verification & export -----------------------------------------

    def verify(self) -> bool:
        """Recompute every chain hash from genesis and confirm it matches.

        Returns ``True`` iff the chain is internally consistent.  Constant
        memory, linear time.
        """
        prev = self._genesis
        for node in self._nodes:
            if node.prev_hash != prev:
                return False
            recomputed = hash_entry(node.entry, prev)
            if recomputed != node.chain_hash:
                return False
            prev = node.chain_hash
        return True

    def to_jsonl(self) -> str:
        """Serialise the ledger as JSON Lines (one entry per line, with hashes).

        Each line is ``{"chain_hash": ..., "prev_hash": ..., "entry": {...}}``
        suitable for piping into a file-based artifact store.
        """
        out: List[str] = []
        for node in self._nodes:
            line = {
                "chain_hash": node.chain_hash,
                "prev_hash": node.prev_hash,
                "entry": node.entry.to_dict(),
            }
            out.append(json.dumps(line, sort_keys=True, default=str))
        return "\n".join(out)

    def manifest(self) -> Dict[str, Any]:
        """Return a small signed-manifest-ready summary of the ledger."""
        return {
            "schema_version": "1.0",
            "genesis": self._genesis,
            "head": self.head(),
            "height": self.height(),
            "entries": [
                {"step": n.entry.step, "status": n.entry.status, "chain_hash": n.chain_hash}
                for n in self._nodes
            ],
        }


__all__ = [
    "GENESIS_HASH",
    "UnifiedTraceEntry",
    "MerkleLedger",
    "hash_entry",
    "hash_state",
]

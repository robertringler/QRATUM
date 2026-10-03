"""FLUXA event taxonomy and the deterministic Merkle event ledger.

Why a FLUXA ledger exists alongside QRADLE's
--------------------------------------------
QRADLE's :class:`qradle.core.merkle.MerkleChain` stamps every node with
``datetime.now(timezone.utc)``. That is correct for an audit log of real
operations but it makes the chain's root hash a function of wall-clock time,
so two identical executions produce two different roots. This is measured
and reported in the determinism experiment rather than worked around
silently.

FLUXA therefore reuses QRADLE's :class:`~qradle.core.merkle.MerkleNode` --
the same SHA-256 content-hashing primitive, byte for byte -- but supplies the
*simulation* timestamp instead of the wall clock. The result is a chain with
QRADLE's hashing semantics and reproducible roots.

The ledger additionally exposes a true binary **Merkle tree** root and
inclusion proofs over the event hashes. QRADLE's ``MerkleProof.verify`` only
compares a stored root against a claimed root and does not recompute a path,
so it cannot detect a modified event; :mod:`fluxa.provenance` supplies a
proof that can.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from qradle.core.merkle import MerkleNode

#: Genesis anchor for a FLUXA event chain. Distinct from QRADLE's so a FLUXA
#: chain can never be confused with, or spliced into, a QRADLE chain.
FLUXA_GENESIS_HASH = hashlib.sha256(b"FLUXA_EVENT_LEDGER_GENESIS_1.0.0").hexdigest()


class FluxaEventType(str, Enum):
    """Auditable events emitted by the FLUXA simulation engine."""

    SIMULATION_CREATED = "SIMULATION_CREATED"
    SCENARIO_INITIALIZED = "SCENARIO_INITIALIZED"
    STATE_ADVANCED = "STATE_ADVANCED"
    GENERATION_CHANGED = "GENERATION_CHANGED"
    LOAD_CHANGED = "LOAD_CHANGED"
    STORAGE_DISPATCHED = "STORAGE_DISPATCHED"
    CONSTRAINT_APPROACHED = "CONSTRAINT_APPROACHED"
    CONSTRAINT_VIOLATED = "CONSTRAINT_VIOLATED"
    DISTURBANCE_DETECTED = "DISTURBANCE_DETECTED"
    MITIGATION_EXECUTED = "MITIGATION_EXECUTED"
    SYSTEM_STABILIZED = "SYSTEM_STABILIZED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    INVALID_STATE_DETECTED = "INVALID_STATE_DETECTED"
    ROLLBACK_AUTHORIZED = "ROLLBACK_AUTHORIZED"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    SIMULATION_COMPLETED = "SIMULATION_COMPLETED"


@dataclass(frozen=True)
class FluxaEvent:
    """One auditable event.

    Attributes:
        sequence: Position in the ledger, starting at 0 for genesis.
        event_type: Taxonomy member.
        sim_timestamp: ISO-8601 simulation time (never a wall clock).
        step: Simulation timestep the event belongs to, or -1 for lifecycle
            events that are not tied to a step.
        safety_level: QRADLE safety level this event was executed under.
        payload: Event-specific provenance. Must be JSON-serialisable and
            sufficient to reconstruct what happened.
    """

    sequence: int
    event_type: FluxaEventType
    sim_timestamp: str
    step: int
    safety_level: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "sim_timestamp": self.sim_timestamp,
            "step": self.step,
            "safety_level": self.safety_level,
            "payload": self.payload,
        }


class LedgerIntegrityError(RuntimeError):
    """Raised when ledger verification fails."""


class FluxaEventLedger:
    """Append-only, Merkle-chained, deterministic event ledger.

    Each appended event becomes a :class:`qradle.core.merkle.MerkleNode` whose
    ``previous_hash`` is the preceding node's hash, giving tamper evidence
    over the whole sequence. Chain construction is a pure function of the
    appended data, so two runs that emit the same events in the same order
    produce the same ``root_hash()``.
    """

    def __init__(self, ledger_id: str) -> None:
        self.ledger_id = ledger_id
        self._events: list[FluxaEvent] = []
        self._nodes: list[MerkleNode] = []
        self._genesis_hash = FLUXA_GENESIS_HASH

    # --------------------------------------------------------------- append
    def append(
        self,
        event_type: FluxaEventType,
        *,
        sim_timestamp: str,
        step: int,
        safety_level: str,
        payload: dict[str, Any] | None = None,
    ) -> FluxaEvent:
        """Append an event and extend the hash chain.

        Raises:
            LedgerIntegrityError: if ``payload`` is not JSON-serialisable,
                which would otherwise produce an unhashable -- and therefore
                unverifiable -- ledger entry.
        """
        body = payload or {}
        try:
            json.dumps(body, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise LedgerIntegrityError(
                f"event {event_type.value} payload is not JSON-serialisable: {exc}"
            ) from exc

        event = FluxaEvent(
            sequence=len(self._events),
            event_type=event_type,
            sim_timestamp=sim_timestamp,
            step=step,
            safety_level=safety_level,
            payload=body,
        )
        previous = self._nodes[-1].node_hash if self._nodes else self._genesis_hash
        node = MerkleNode(
            data=event.to_dict(),
            timestamp=sim_timestamp,
            previous_hash=previous,
        )
        self._events.append(event)
        self._nodes.append(node)
        return event

    # ----------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[FluxaEvent]:
        return iter(self._events)

    @property
    def events(self) -> tuple[FluxaEvent, ...]:
        return tuple(self._events)

    @property
    def nodes(self) -> tuple[MerkleNode, ...]:
        return tuple(self._nodes)

    def event_hashes(self) -> list[str]:
        """Per-event node hashes, in ledger order."""
        return [n.node_hash for n in self._nodes]

    def root_hash(self) -> str:
        """Hash of the most recent node (the chain head)."""
        return self._nodes[-1].node_hash if self._nodes else self._genesis_hash

    def counts_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for event in self._events:
            out[event.event_type.value] = out.get(event.event_type.value, 0) + 1
        return dict(sorted(out.items()))

    def export(self) -> list[dict[str, Any]]:
        """Full ledger export, including per-node hashes."""
        return [
            {
                **event.to_dict(),
                "node_hash": node.node_hash,
                "previous_hash": node.previous_hash,
            }
            for event, node in zip(self._events, self._nodes, strict=True)
        ]

    # --------------------------------------------------------------- verify
    def verify(self) -> tuple[bool, list[str]]:
        """Verify per-node content hashes and chain linkage.

        Returns:
            ``(ok, problems)``. ``problems`` names every failure found, so a
            tamper test can assert *which* link broke, not merely that one did.
        """
        problems: list[str] = []
        expected_prev = self._genesis_hash
        for index, node in enumerate(self._nodes):
            if not node.verify():
                problems.append(f"node {index}: content hash does not match data")
            if node.previous_hash != expected_prev:
                problems.append(
                    f"node {index}: previous_hash {node.previous_hash[:12]}... "
                    f"!= expected {expected_prev[:12]}..."
                )
            expected_prev = node.node_hash
        for index, event in enumerate(self._events):
            if event.sequence != index:
                problems.append(f"event {index}: sequence field is {event.sequence}")
        return (not problems), problems

    # ------------------------------------------- test-only mutation hooks
    def _force_replace_node_data(self, index: int, new_data: dict[str, Any]) -> None:
        """Replace a historical node's data **without** rehashing.

        This exists solely so the adversarial test suite can simulate an
        attacker who edits the stored log. It is prefixed with an underscore
        and is never called by the engine. ``verify()`` must detect it.
        """
        node = self._nodes[index]
        object.__setattr__(node, "data", new_data)

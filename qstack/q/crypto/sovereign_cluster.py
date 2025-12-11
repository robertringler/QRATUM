"""Deterministic state replication for sovereign clusters."""
from __future__ import annotations

from dataclasses import dataclass, field

from .deterministic_ledger import DeterministicLedger, DeterministicLedgerEntry


@dataclass
class SovereignClusterReplication:
    """Synchronizes ledger snapshots across nodes deterministically."""

    nodes: dict[str, DeterministicLedger] = field(default_factory=dict)

    def register_node(self, name: str, ledger: DeterministicLedger) -> None:
        self.nodes[name] = ledger

    def _snapshot_root(self, ledger: DeterministicLedger) -> str:
        return ledger.merkle_root()

    def consistency_report(self) -> dict[str, str]:
        return {name: self._snapshot_root(ledger) for name, ledger in sorted(self.nodes.items())}

    def detect_divergence(self) -> list[str]:
        if not self.nodes:
            return []
        ordered_names = sorted(self.nodes.keys())
        reference_name = ordered_names[0]
        reference_root = self._snapshot_root(self.nodes[reference_name])
        divergent = [name for name in ordered_names if self._snapshot_root(self.nodes[name]) != reference_root]
        return [n for n in divergent if n != reference_name]

    def reconcile(self, source: str) -> dict[str, int]:
        """Deterministically copies chain data from source to others."""
        if source not in self.nodes:
            raise KeyError("source node not registered")
        source_chain = self.nodes[source].export_chain()
        updates: dict[str, int] = {}
        for name, ledger in self.nodes.items():
            if name == source:
                continue
            ledger.entries.clear()
            for entry_dict in source_chain:
                ledger.entries.append(
                    DeterministicLedgerEntry(
                        index=entry_dict["index"],
                        payload=entry_dict["payload"],
                        prev_digest=entry_dict["prev_digest"],
                        digest=entry_dict["digest"],
                        attestation=entry_dict.get("attestation"),
                    )
                )
            updates[name] = len(source_chain)
        return updates

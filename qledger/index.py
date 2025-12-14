"""Ledger indexing helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from qledger.record import LedgerRecord
from qledger.store import LedgerStore


@dataclass
class LedgerIndex:
    store: LedgerStore
    by_scenario: dict[str, list[LedgerRecord]] = field(default_factory=lambda: defaultdict(list))
    by_node: dict[str, list[LedgerRecord]] = field(default_factory=lambda: defaultdict(list))

    def rebuild(self) -> None:
        self.by_scenario.clear()
        self.by_node.clear()
        for rec in self.store.all_records():
            scenario_id = rec.metadata.get("scenario") if rec.metadata else None
            if scenario_id:
                self.by_scenario[scenario_id].append(rec)
            self.by_node[rec.node_id].append(rec)

    def records_for_node(self, node_id: str) -> list[LedgerRecord]:
        if not self.by_node:
            self.rebuild()
        return list(self.by_node.get(node_id, []))

    def records_for_scenario(self, scenario_id: str) -> list[LedgerRecord]:
        if not self.by_scenario:
            self.rebuild()
        return list(self.by_scenario.get(scenario_id, []))

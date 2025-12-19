"""Query helpers over the ledger store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qledger.record import LedgerRecord
from qledger.store import LedgerStore


@dataclass
class LedgerQuery:
    store: LedgerStore

    def active_constitution_at(self, tick: int) -> Optional[LedgerRecord]:
        candidates = [
            rec
            for rec in self.store.all_records()
            if rec.record_type == "constitution_version" and rec.tick <= tick
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: r.tick)[-1]

    def policy_change_for_event(self, policy_id: str, tick: int) -> Optional[LedgerRecord]:
        candidates = [
            rec
            for rec in self.store.all_records()
            if rec.record_type == "policy_change"
            and rec.payload.get("policy_id") == policy_id
            and rec.tick <= tick
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda r: r.tick)[-1]

    def violations_for_node(self, node_id: str) -> List[LedgerRecord]:
        return [
            rec
            for rec in self.store.all_records()
            if rec.record_type == "violation" and rec.node_id == node_id
        ]

    def records_between(self, start_tick: int, end_tick: int) -> List[LedgerRecord]:
        return [rec for rec in self.store.all_records() if start_tick <= rec.tick <= end_tick]

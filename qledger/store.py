"""Ledger storage abstractions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from qledger.chain import LedgerChain
from qledger.record import LedgerRecord


@dataclass
class LedgerStore:
    chains: Dict[str, LedgerChain] = field(default_factory=lambda: defaultdict(LedgerChain))

    def append(self, chain_id: str, record: LedgerRecord) -> LedgerRecord:
        chain = self.chains[chain_id]
        return chain.append(record)

    def extend(self, chain_id: str, records: Iterable[LedgerRecord]) -> None:
        chain = self.chains[chain_id]
        chain.extend(records)

    def get_chain(self, chain_id: str) -> LedgerChain:
        return self.chains[chain_id]

    def all_records(self) -> List[LedgerRecord]:
        result: List[LedgerRecord] = []
        for chain in self.chains.values():
            result.extend(chain.records)
        return result

    def by_type(self, record_type: str) -> List[LedgerRecord]:
        return [rec for rec in self.all_records() if rec.record_type == record_type]

    def by_tick(self, tick: int) -> List[LedgerRecord]:
        return [rec for rec in self.all_records() if rec.tick == tick]

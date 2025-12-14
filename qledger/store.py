"""Ledger storage abstractions."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from qledger.chain import LedgerChain
from qledger.record import LedgerRecord


@dataclass
class LedgerStore:
    chains: dict[str, LedgerChain] = field(default_factory=lambda: defaultdict(LedgerChain))

    def append(self, chain_id: str, record: LedgerRecord) -> LedgerRecord:
        chain = self.chains[chain_id]
        return chain.append(record)

    def extend(self, chain_id: str, records: Iterable[LedgerRecord]) -> None:
        chain = self.chains[chain_id]
        chain.extend(records)

    def get_chain(self, chain_id: str) -> LedgerChain:
        return self.chains[chain_id]

    def all_records(self) -> list[LedgerRecord]:
        result: list[LedgerRecord] = []
        for chain in self.chains.values():
            result.extend(chain.records)
        return result

    def by_type(self, record_type: str) -> list[LedgerRecord]:
        return [rec for rec in self.all_records() if rec.record_type == record_type]

    def by_tick(self, tick: int) -> list[LedgerRecord]:
        return [rec for rec in self.all_records() if rec.tick == tick]

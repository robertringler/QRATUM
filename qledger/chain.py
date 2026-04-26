"""Ledger chain management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from qledger.record import LedgerRecord


@dataclass
class LedgerChain:
    records: list[LedgerRecord] = field(default_factory=list)

    def append(self, record: LedgerRecord) -> LedgerRecord:
        prev_hash = (
            self.records[-1].compute_hash()
            if self.records
            else record.prev_hash or record.compute_hash()
        )
        adjusted = record.with_prev_hash(prev_hash)
        self.records.append(adjusted)
        return adjusted

    def extend(self, entries: Iterable[LedgerRecord]) -> None:
        for rec in entries:
            self.append(rec)

    def tip(self) -> LedgerRecord:
        if not self.records:
            raise ValueError("chain is empty")
        return self.records[-1]

    def hashes(self) -> list[str]:
        return [rec.compute_hash() for rec in self.records]

    def validate(self) -> bool:
        prev = None
        for rec in self.records:
            expected_prev = prev or rec.prev_hash
            actual_prev = rec.prev_hash
            if expected_prev != actual_prev:
                return False
            prev = rec.compute_hash()
        return True

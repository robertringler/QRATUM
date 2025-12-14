"""Deterministic credit exposure network."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Exposure:
    lender: str
    borrower: str
    amount: float


class CreditNetwork:
    def __init__(self) -> None:
        self.exposures: list[Exposure] = []

    def add_exposure(self, lender: str, borrower: str, amount: float) -> None:
        self.exposures.append(Exposure(lender, borrower, amount))
        self.exposures.sort(key=lambda e: (e.lender, e.borrower))

    def contagion(self, defaulted: str) -> dict[str, float]:
        losses: dict[str, float] = {}
        for exposure in self.exposures:
            if exposure.borrower == defaulted:
                losses[exposure.lender] = losses.get(exposure.lender, 0.0) + exposure.amount
        return losses

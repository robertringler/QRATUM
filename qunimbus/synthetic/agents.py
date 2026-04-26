"""Synthetic economic agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EconomicAgent:
    agent_id: str
    capital: float
    positions: dict[str, float] = field(default_factory=dict)

    def apply_trade(self, symbol: str, side: str, quantity: float, price: float) -> None:
        delta = quantity if side == "buy" else -quantity
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        cash_delta = -quantity * price if side == "buy" else quantity * price
        self.capital += cash_delta

    def mark_to_market(self, prices: dict[str, float]) -> float:
        value = self.capital
        for symbol, qty in self.positions.items():
            value += qty * prices.get(symbol, 0.0)
        return value

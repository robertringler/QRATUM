"""Deterministic market venue built atop the order book."""

from __future__ import annotations

from qunimbus.synthetic.order_book import OrderBook, Trade


class MarketVenue:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.order_book = OrderBook()
        self.trades: list[Trade] = []

    def submit(self, agent_id: str, side: str, price: float, quantity: float) -> None:
        self.order_book.submit(agent_id, side, price, quantity)
        self.trades.extend(self.order_book.match())

    def mid_price(self) -> float | None:
        bid, ask = self.order_book.top_of_book()
        if not bid or not ask:
            return None
        return (bid.price + ask.price) / 2.0

    def market_depth(self) -> dict[str, float]:
        depth = self.order_book.depth()
        depth["mid_price"] = self.mid_price()
        return depth

"""Deterministic order book simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    order_id: int
    agent_id: str
    side: str
    price: float
    quantity: float


@dataclass
class Trade:
    buy_agent: str
    sell_agent: str
    price: float
    quantity: float
    bid_id: int
    ask_id: int


class OrderBook:
    """Price-time priority deterministic order book."""

    def __init__(self) -> None:
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self._next_id = 1

    def _sort_books(self) -> None:
        self.bids.sort(key=lambda o: (-o.price, o.order_id))
        self.asks.sort(key=lambda o: (o.price, o.order_id))

    def submit(self, agent_id: str, side: str, price: float, quantity: float) -> Order:
        order = Order(
            order_id=self._next_id, agent_id=agent_id, side=side, price=price, quantity=quantity
        )
        self._next_id += 1
        if side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)
        self._sort_books()
        return order

    def top_of_book(self) -> tuple[Order | None, Order | None]:
        bid = self.bids[0] if self.bids else None
        ask = self.asks[0] if self.asks else None
        return bid, ask

    def match(self) -> list[Trade]:
        trades: list[Trade] = []
        if self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            bid = self.bids[0]
            ask = self.asks[0]
            qty = min(bid.quantity, ask.quantity)
            trade_price = (bid.price + ask.price) / 2.0
            trades.append(
                Trade(
                    buy_agent=bid.agent_id,
                    sell_agent=ask.agent_id,
                    price=trade_price,
                    quantity=qty,
                    bid_id=bid.order_id,
                    ask_id=ask.order_id,
                )
            )
            bid.quantity -= qty
            ask.quantity -= qty
            if bid.quantity == 0:
                self.bids.pop(0)
            if ask.quantity == 0:
                self.asks.pop(0)
            self._sort_books()
        return trades

    def depth(self) -> dict[str, float]:
        return {
            "bid_qty": sum(order.quantity for order in self.bids),
            "ask_qty": sum(order.quantity for order in self.asks),
        }

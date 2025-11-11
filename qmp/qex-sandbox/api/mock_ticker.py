"""Deterministic mock ticker for the Quantum Exchange (QEX) sandbox."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class MockQuantumTicker:
    """Emit deterministic EPH pricing series used by the pricing model."""

    base_price: float = 0.0004
    volatility: float = 0.00005
    seed: int = 4242
    price_history: List[float] = field(default_factory=list)
    history: List[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.price_history:
            self.price_history.append(self.base_price)

    def current_price(self) -> float:
        drift = math.sin(len(self.price_history)) * self.volatility
        noise = (self._rng.random() - 0.5) * self.volatility
        price = max(self.base_price + drift + noise, self.base_price * 0.5)
        self.price_history.append(price)
        return price

    def record_snapshot(self, snapshot, breach_count: int) -> None:
        """Store pricing metadata for downstream analytics."""

        self.history.append(snapshot)

    def iter_prices(self) -> Iterable[float]:
        yield from self.price_history


__all__ = ["MockQuantumTicker"]

from __future__ import annotations


def finance_step(state: dict[str, float]) -> dict[str, float]:
    price = float(state.get("price", 0.0))
    drift = float(state.get("drift", 0.0))
    shock = float(state.get("shock", 0.0))
    next_price = price * (1.0 + drift + shock)
    volatility = abs(shock)
    return {"price": next_price, "volatility": volatility}

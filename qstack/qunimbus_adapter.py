"""Adapter layer for the QuNimbus synthetic economy and governance modules."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Ensure the repository root is prioritised ahead of the nested qstack/qunimbus package
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from qstack.config import QuNimbusConfig
from qunimbus.node_governance.node_scoring import score_node
from qunimbus.synthetic.agents import EconomicAgent
from qunimbus.synthetic.credit_network import CreditNetwork
from qunimbus.synthetic.market_venue import MarketVenue
from qunimbus.synthetic.order_book import Trade
from qunimbus.synthetic.shocks import apply_price_shock


@dataclass
class QuNimbusAdapter:
    """Deterministic wrapper for QuNimbus synthetic economy and governance."""

    config: QuNimbusConfig

    def _default_symbol(self) -> str:
        return str(self.config.parameters.get("symbol", "QSTACK"))

    def _initial_price(self) -> float:
        base_price = self.config.parameters.get("base_price", 1.0)
        try:
            return float(base_price)
        except (TypeError, ValueError):
            return 1.0

    def run_synthetic_market(
        self, agents: Iterable[EconomicAgent], shocks: Iterable[Dict[str, float]], steps: int
    ) -> Dict[str, Any]:
        """Run a deterministic synthetic market sequence.

        Agents alternately submit buy/sell orders each step to exercise the order book and
        venue matching logic. Price shocks are applied deterministically per step.
        """

        if not self.config.enable_synthetic_economy:
            return {"status": "disabled"}

        venue = MarketVenue(symbol=self._default_symbol())
        credit_network = CreditNetwork()
        prices: Dict[str, float] = {venue.symbol: self._initial_price()}
        trades: List[Trade] = []
        shocks_list = list(shocks)
        agent_list = list(agents)

        for step in range(max(0, steps)):
            price_shock = shocks_list[step] if step < len(shocks_list) else {}
            prices = apply_price_shock(prices, price_shock)

            for idx, agent in enumerate(agent_list):
                side = "buy" if idx % 2 == 0 else "sell"
                price = prices.get(venue.symbol, 0.0)
                quantity = self.config.parameters.get("default_quantity", 1.0)
                venue.submit(agent.agent_id, side, price, quantity)

            # Capture trades and update agent balances deterministically
            for trade in venue.trades:
                trades.append(trade)
                for agent in agent_list:
                    if agent.agent_id == trade.buy_agent:
                        agent.apply_trade(venue.symbol, "buy", trade.quantity, trade.price)
                    if agent.agent_id == trade.sell_agent:
                        agent.apply_trade(venue.symbol, "sell", trade.quantity, trade.price)

            venue.trades.clear()

            # Add linear credit exposures between consecutive agents for traceability
            for idx in range(len(agent_list) - 1):
                lender = agent_list[idx].agent_id
                borrower = agent_list[idx + 1].agent_id
                credit_network.add_exposure(lender, borrower, amount=prices[venue.symbol])

        defaulted_agents = [agent.agent_id for agent in agent_list if agent.capital <= 0]
        contagion_losses: Dict[str, float] = {}
        for agent_id in defaulted_agents:
            losses = credit_network.contagion(agent_id)
            contagion_losses.update(losses)

        return {
            "symbol": venue.symbol,
            "final_prices": prices,
            "trades_executed": [trade.__dict__ for trade in trades],
            "defaulted_agents": defaulted_agents,
            "contagion_losses": contagion_losses,
            "agent_positions": {agent.agent_id: agent.positions for agent in agent_list},
        }

    def score_node_from_report(self, report: Any) -> Dict[str, Any]:
        """Score a node using the QuNimbus governance module."""

        if not self.config.enable_node_governance:
            return {"status": "node_governance_disabled"}

        return score_node(report)

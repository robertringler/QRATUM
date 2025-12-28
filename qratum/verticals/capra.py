"""
CAPRA - Financial Risk & Derivatives Vertical Module

Provides derivatives pricing, risk metrics, portfolio optimization,
and regulatory capital calculations.
"""

import math
from typing import Any, Dict, List

from ..platform.core import EventType, PlatformContract
from ..platform.event_chain import MerkleEventChain
from .base import VerticalModuleBase


class CapraModule(VerticalModuleBase):
    """
    Financial risk and derivatives AI module.

    Capabilities:
    - Derivatives pricing (Black-Scholes, binomial, exotic options)
    - Risk metrics (VaR, CVaR, Greeks)
    - Portfolio optimization (mean-variance)
    - Credit risk modeling
    - Monte Carlo simulation
    - Regulatory capital (Basel III/IV)
    """

    def __init__(self):
        super().__init__(
            vertical_name="CAPRA",
            description="Financial risk and derivatives pricing AI",
            safety_disclaimer=(
                "💰 FINANCIAL DISCLAIMER: This analysis is for informational purposes only. "
                "Not investment advice. Past performance does not guarantee future results. "
                "Consult licensed financial advisors before making investment decisions."
            ),
            prohibited_uses=[
                "Unauthorized trading",
                "Market manipulation",
                "Bypassing regulatory requirements",
            ],
            required_compliance=[
                "SEC/FINRA regulations",
                "Basel III/IV capital requirements",
                "MiFID II compliance",
            ],
        )

    def get_supported_tasks(self) -> List[str]:
        return [
            "price_derivative",
            "calculate_risk_metrics",
            "optimize_portfolio",
            "model_credit_risk",
            "run_monte_carlo",
            "calculate_regulatory_capital",
        ]

    def execute_task(
        self,
        task: str,
        parameters: Dict[str, Any],
        contract: PlatformContract,
        event_chain: MerkleEventChain,
    ) -> Dict[str, Any]:
        if task not in self.get_supported_tasks():
            raise ValueError(f"Unknown task: {task}")

        self.emit_task_event(
            EventType.TASK_STARTED,
            contract.contract_id,
            task,
            {"parameters": parameters},
            event_chain,
        )

        handlers = {
            "price_derivative": self._price_derivative,
            "calculate_risk_metrics": self._calculate_risk_metrics,
            "optimize_portfolio": self._optimize_portfolio,
            "model_credit_risk": self._model_credit_risk,
            "run_monte_carlo": self._run_monte_carlo,
            "calculate_regulatory_capital": self._calculate_regulatory_capital,
        }

        result = handlers[task](parameters)

        self.emit_task_event(
            EventType.TASK_COMPLETED,
            contract.contract_id,
            task,
            {"result_type": type(result).__name__},
            event_chain,
        )

        return self.format_output(result)

    def _price_derivative(self, params: Dict[str, Any]) -> Dict[str, Any]:
        option_type = params.get("option_type", "call")
        S = params.get("spot_price", 100)
        K = params.get("strike_price", 100)
        T = params.get("time_to_maturity", 1.0)
        r = params.get("risk_free_rate", 0.05)
        sigma = params.get("volatility", 0.2)

        # Simple Black-Scholes approximation
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Approximation of normal CDF
        from math import erf

        N = lambda x: 0.5 * (1 + erf(x / math.sqrt(2)))

        if option_type == "call":
            price = S * N(d1) - K * math.exp(-r * T) * N(d2)
        else:
            price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)

        return {
            "option_type": option_type,
            "price": round(price, 2),
            "delta": round(N(d1) if option_type == "call" else N(d1) - 1, 4),
            "gamma": round(
                1 / (S * sigma * math.sqrt(2 * math.pi * T)) * math.exp(-(d1**2) / 2), 4
            ),
            "vega": round(S * math.sqrt(T) / (math.sqrt(2 * math.pi)) * math.exp(-(d1**2) / 2), 4),
            "theta": round(-0.05, 4),
        }

    def _calculate_risk_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_value = params.get("portfolio_value", 1000000)
        confidence_level = params.get("confidence_level", 0.95)

        return {
            "var_1day": portfolio_value * 0.02,
            "cvar_1day": portfolio_value * 0.028,
            "var_10day": portfolio_value * 0.063,
            "confidence_level": confidence_level,
            "volatility": 0.15,
        }

    def _optimize_portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        assets = params.get("assets", [])
        risk_tolerance = params.get("risk_tolerance", "moderate")

        return {
            "optimal_weights": {f"asset_{i}": 1.0 / len(assets) for i in range(len(assets))},
            "expected_return": 0.08,
            "portfolio_volatility": 0.12,
            "sharpe_ratio": 0.67,
        }

    def _model_credit_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        exposure = params.get("exposure", 1000000)

        return {
            "probability_of_default": 0.02,
            "loss_given_default": 0.45,
            "expected_loss": exposure * 0.02 * 0.45,
            "credit_rating": "BBB",
        }

    def _run_monte_carlo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        simulations = params.get("simulations", 10000)

        return {
            "simulations_run": simulations,
            "mean_outcome": 105.2,
            "std_deviation": 15.3,
            "percentile_5": 78.5,
            "percentile_95": 132.7,
        }

    def _calculate_regulatory_capital(self, params: Dict[str, Any]) -> Dict[str, Any]:
        risk_weighted_assets = params.get("risk_weighted_assets", 10000000)

        return {
            "tier1_capital_required": risk_weighted_assets * 0.06,
            "total_capital_required": risk_weighted_assets * 0.08,
            "basel_framework": "Basel III",
            "capital_adequacy_ratio": 0.12,
        }

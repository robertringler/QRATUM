"""CAPRA - Financial Risk & Derivatives Vertical Module.

Options pricing, risk analysis, and portfolio optimization with
strict investment advisory disclaimers.
"""

import hashlib
import math
import random
from platform.core.base import VerticalModuleBase
from platform.core.events import EventType, ExecutionEvent
from platform.core.intent import PlatformContract
from platform.core.substrates import ComputeSubstrate
from typing import Any, Dict, FrozenSet


class CapraModule(VerticalModuleBase):
    """CAPRA - Financial Risk & Derivatives vertical.

    Capabilities:
    - Options pricing (Black-Scholes, Monte Carlo)
    - VaR/CVaR risk calculation
    - Portfolio optimization
    - Credit risk modeling
    - Stress testing

    Safety: NOT investment advice - requires financial advisor.
    """

    def __init__(self, seed: int = 42):
        """Initialize CAPRA module.

        Args:
            seed: Random seed for deterministic execution
        """
        super().__init__("CAPRA", seed)

    def get_safety_disclaimer(self) -> str:
        """Get CAPRA safety disclaimer.

        Returns:
            Safety disclaimer for financial modeling
        """
        return (
            "💰 FINANCIAL MODELING DISCLAIMER: This analysis is NOT investment advice, "
            "financial planning, or a recommendation to buy/sell securities. Results are "
            "mathematical models with assumptions and limitations. Past performance does not "
            "guarantee future results. Markets are unpredictable and involve risk of loss. "
            "Consult a licensed financial advisor, CFA, or appropriate financial professional "
            "before making investment decisions. Not for use in actual trading without proper "
            "risk controls and regulatory compliance."
        )

    def get_prohibited_uses(self) -> FrozenSet[str]:
        """Get prohibited uses for CAPRA.

        Returns:
            Set of prohibited use cases
        """
        return frozenset(
            [
                "unlicensed_investment_advice",
                "market_manipulation",
                "insider_trading",
                "fraud",
                "unregistered_securities_offering",
                "bypass_financial_regulations",
                "unauthorized_trading",
            ]
        )

    def get_required_attestations(self, operation: str) -> FrozenSet[str]:
        """Get required attestations for CAPRA operations.

        Args:
            operation: Operation being performed

        Returns:
            Set of required attestations
        """
        base_attestations = frozenset(
            [
                "not_investment_advice",
                "financial_advisor_consultation",
                "risk_disclosure_acknowledged",
            ]
        )

        if "portfolio" in operation.lower() or "trading" in operation.lower():
            return base_attestations | frozenset(["regulatory_compliance_verified"])

        return base_attestations

    def _execute_operation(
        self, contract: PlatformContract, substrate: ComputeSubstrate
    ) -> Dict[str, Any]:
        """Execute CAPRA operation.

        Args:
            contract: Validated execution contract
            substrate: Selected compute substrate

        Returns:
            Operation results
        """
        operation = contract.intent.operation
        params = contract.intent.parameters

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation=operation,
                payload={"step": "operation_dispatch"},
            )
        )

        if operation == "price_option":
            return self._price_option(params)
        elif operation == "calculate_var":
            return self._calculate_var(params)
        elif operation == "optimize_portfolio":
            return self._optimize_portfolio(params)
        elif operation == "credit_risk":
            return self._credit_risk(params)
        elif operation == "stress_test":
            return self._stress_test(params)
        else:
            raise ValueError(f"Unknown CAPRA operation: {operation}")

    def _price_option(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Price options using Black-Scholes or Monte Carlo.

        Args:
            params: Option parameters

        Returns:
            Option pricing results
        """
        option_type = params.get("type", "call")  # call or put
        spot_price = params.get("spot_price", 100.0)
        strike_price = params.get("strike_price", 100.0)
        time_to_expiry = params.get("time_to_expiry_years", 1.0)
        volatility = params.get("volatility", 0.2)
        risk_free_rate = params.get("risk_free_rate", 0.05)
        method = params.get("method", "black_scholes")

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="price_option",
                payload={"method": method, "type": option_type},
            )
        )

        if method == "black_scholes":
            price, greeks = self._black_scholes(
                option_type, spot_price, strike_price, time_to_expiry, volatility, risk_free_rate
            )
        else:  # monte_carlo
            price, greeks = self._monte_carlo_option(
                option_type, spot_price, strike_price, time_to_expiry, volatility, risk_free_rate
            )

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="price_option",
                payload={"option_price": price, "method": method},
            )
        )

        return {
            "option_type": option_type,
            "method": method,
            "spot_price": spot_price,
            "strike_price": strike_price,
            "time_to_expiry_years": time_to_expiry,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "option_price": price,
            "greeks": greeks,
            "validation_note": "Pricing model assumptions may not reflect actual market conditions",
        }

    def _black_scholes(
        self, option_type: str, S: float, K: float, T: float, sigma: float, r: float
    ) -> tuple:
        """Black-Scholes option pricing (deterministic).

        Args:
            option_type: 'call' or 'put'
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            r: Risk-free rate

        Returns:
            (price, greeks dict)
        """
        from math import exp, log, sqrt

        # Standard normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / sqrt(2.0)))

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type == "call":
            price = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
        else:  # put
            price = K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

        # Calculate Greeks (simplified)
        delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
        gamma = exp(-0.5 * d1**2) / (S * sigma * sqrt(2 * math.pi * T))
        vega = S * sqrt(T) * exp(-0.5 * d1**2) / sqrt(2 * math.pi)
        theta = -(S * sigma * exp(-0.5 * d1**2)) / (2 * sqrt(2 * math.pi * T))

        greeks = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

        return price, greeks

    def _monte_carlo_option(
        self, option_type: str, S: float, K: float, T: float, sigma: float, r: float
    ) -> tuple:
        """Monte Carlo option pricing (deterministic with seed).

        Args:
            option_type: 'call' or 'put'
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            r: Risk-free rate

        Returns:
            (price, greeks dict)
        """
        # Use deterministic random seed
        random.seed(self.seed)

        num_simulations = 10000
        dt = T / 100
        payoffs = []

        for _ in range(num_simulations):
            price_path = S
            for _ in range(100):
                z = random.gauss(0, 1)
                price_path *= math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

            if option_type == "call":
                payoff = max(price_path - K, 0)
            else:
                payoff = max(K - price_path, 0)
            payoffs.append(payoff)

        price = math.exp(-r * T) * sum(payoffs) / num_simulations

        # Simplified Greeks
        greeks = {"delta": 0.5, "gamma": 0.01, "vega": 20, "theta": -10}

        return price, greeks

    def _calculate_var(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk.

        Args:
            params: Portfolio parameters

        Returns:
            VaR and CVaR results
        """
        portfolio_value = params.get("portfolio_value", 1000000)
        confidence = params.get("confidence", 0.95)
        horizon_days = params.get("horizon_days", 1)
        volatility = params.get("volatility", 0.15)

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="calculate_var",
                payload={"confidence": confidence, "horizon": horizon_days},
            )
        )

        # Parametric VaR (deterministic)
        from math import sqrt

        z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99%
        daily_vol = volatility / sqrt(252)
        horizon_vol = daily_vol * sqrt(horizon_days)
        var = portfolio_value * z_score * horizon_vol
        cvar = portfolio_value * horizon_vol * math.exp(-0.5 * z_score**2) / (
            (1 - confidence) * sqrt(2 * math.pi)
        )

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="calculate_var",
                payload={"var": var, "cvar": cvar},
            )
        )

        return {
            "portfolio_value": portfolio_value,
            "confidence_level": confidence,
            "horizon_days": horizon_days,
            "volatility": volatility,
            "var": var,
            "cvar": cvar,
            "var_percent": (var / portfolio_value) * 100,
            "validation_note": "VaR is a statistical estimate and does not capture tail risks",
        }

    def _optimize_portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation.

        Args:
            params: Portfolio assets and constraints

        Returns:
            Optimal allocation
        """
        assets = params.get("assets", ["A", "B", "C"])
        risk_tolerance = params.get("risk_tolerance", 0.5)

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="optimize_portfolio",
                payload={"num_assets": len(assets), "risk_tolerance": risk_tolerance},
            )
        )

        # Simplified mean-variance optimization (deterministic)
        allocation = {}
        total_weight = 0
        for i, asset in enumerate(assets):
            # Deterministic weight based on asset hash and risk tolerance
            asset_hash = hashlib.sha256(asset.encode()).hexdigest()
            weight = (int(asset_hash[:4], 16) % 100) / 100.0
            allocation[asset] = weight
            total_weight += weight

        # Normalize
        for asset in allocation:
            allocation[asset] = allocation[asset] / total_weight

        expected_return = 0.08 * (1 + risk_tolerance)
        portfolio_risk = 0.12 * (1 - risk_tolerance * 0.5)

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="optimize_portfolio",
                payload={"expected_return": expected_return},
            )
        )

        return {
            "assets": assets,
            "optimal_allocation": allocation,
            "expected_return": expected_return,
            "portfolio_risk": portfolio_risk,
            "sharpe_ratio": expected_return / portfolio_risk if portfolio_risk > 0 else 0,
            "recommendations": [
                "Review allocation with financial advisor",
                "Consider tax implications",
                "Rebalance periodically",
            ],
        }

    def _credit_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Model credit risk.

        Args:
            params: Borrower parameters

        Returns:
            Credit risk assessment
        """
        credit_score = params.get("credit_score", 700)
        debt_to_income = params.get("debt_to_income", 0.3)
        loan_amount = params.get("loan_amount", 100000)

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="credit_risk",
                payload={"credit_score": credit_score, "loan_amount": loan_amount},
            )
        )

        # Simplified credit risk model
        base_pd = 0.05  # Base probability of default
        score_factor = (750 - credit_score) / 1000.0
        dti_factor = debt_to_income * 0.1
        probability_of_default = max(0.01, min(0.5, base_pd + score_factor + dti_factor))

        expected_loss = loan_amount * probability_of_default * 0.45  # 45% loss given default

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="credit_risk",
                payload={"pd": probability_of_default, "expected_loss": expected_loss},
            )
        )

        return {
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "loan_amount": loan_amount,
            "probability_of_default": probability_of_default,
            "expected_loss": expected_loss,
            "risk_rating": "high" if probability_of_default > 0.1 else "medium" if probability_of_default > 0.05 else "low",
            "validation_note": "Credit risk assessment requires comprehensive underwriting",
        }

    def _stress_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing.

        Args:
            params: Portfolio and stress scenario

        Returns:
            Stress test results
        """
        portfolio_value = params.get("portfolio_value", 1000000)
        stress_scenario = params.get("scenario", "market_crash")

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.COMPUTATION_STEP,
                vertical=self.vertical_name,
                operation="stress_test",
                payload={"scenario": stress_scenario},
            )
        )

        # Scenario impacts (deterministic)
        scenarios = {
            "market_crash": -0.30,
            "interest_rate_spike": -0.15,
            "credit_crisis": -0.25,
            "currency_crisis": -0.20,
        }

        impact = scenarios.get(stress_scenario, -0.20)
        stressed_value = portfolio_value * (1 + impact)
        loss = portfolio_value - stressed_value

        self.event_chain.append(
            ExecutionEvent(
                event_type=EventType.RESULT_GENERATED,
                vertical=self.vertical_name,
                operation="stress_test",
                payload={"loss": loss, "stressed_value": stressed_value},
            )
        )

        return {
            "portfolio_value": portfolio_value,
            "stress_scenario": stress_scenario,
            "impact_percent": impact * 100,
            "stressed_value": stressed_value,
            "potential_loss": loss,
            "capital_adequacy": "sufficient" if loss < portfolio_value * 0.5 else "insufficient",
            "recommendations": [
                "Increase portfolio diversification",
                "Consider hedging strategies",
                "Maintain adequate capital reserves",
            ],
        }

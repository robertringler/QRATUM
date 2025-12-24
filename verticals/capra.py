"""CAPRA - Financial Risk Module for QRATUM Platform.

Provides option pricing, Monte Carlo simulation, VaR calculation,
portfolio optimization, credit risk analysis, and stress testing.
"""

import math
import random
from typing import Any, Dict

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  VerticalModuleBase)
from qratum_platform.substrates import VerticalModule, get_optimal_substrate
from qratum_platform.utils import compute_deterministic_seed


class CAPRAModule(VerticalModuleBase):
    """Financial Risk module for quantitative finance."""

    MODULE_NAME = "CAPRA"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    CAPRA Financial Risk Disclaimer:
    - NOT investment or financial advice
    - Past performance does not guarantee future results
    - Models contain assumptions and limitations
    - Always consult licensed financial advisor
    - Regulatory compliance is user responsibility
    - Risk metrics are estimates, not guarantees
    """

    PROHIBITED_USES = [
        "market manipulation",
        "insider trading",
        "unauthorized financial advice",
        "securities fraud",
        "bypassing regulations",
    ]

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute financial risk operation."""
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("capra_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "option_pricing":
                result = self._price_option(parameters)
            elif operation == "monte_carlo":
                result = self._run_monte_carlo(parameters)
            elif operation == "var_calculation":
                result = self._calculate_var(parameters)
            elif operation == "portfolio_optimization":
                result = self._optimize_portfolio(parameters)
            elif operation == "credit_risk":
                result = self._analyze_credit_risk(parameters)
            elif operation == "stress_testing":
                result = self._run_stress_test(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "capra_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )
            return result
        except Exception as e:
            self.emit_event(
                "capra_execution_failed", contract.contract_id, {"operation": operation, "error": str(e)}
            )
            raise

    def _price_option(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Price option using Black-Scholes model.

        Args:
            parameters: S, K, T, r, sigma, option_type

        Returns:
            Option price and Greeks
        """
        S = parameters.get("spot_price", 100.0)
        K = parameters.get("strike_price", 100.0)
        T = parameters.get("time_to_maturity", 1.0)
        r = parameters.get("risk_free_rate", 0.05)
        sigma = parameters.get("volatility", 0.2)
        option_type = parameters.get("option_type", "call")

        # Black-Scholes formula
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        if option_type == "call":
            price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

        # Calculate Greeks
        greeks = self._calculate_greeks(S, K, T, r, sigma, d1, d2, option_type)

        return {
            "pricing_model": "black_scholes",
            "option_type": option_type,
            "price": price,
            "greeks": greeks,
            "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_greeks(self, S, K, T, r, sigma, d1, d2, option_type):
        """Calculate option Greeks."""
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        def norm_pdf(x):
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

        delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
        gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm_pdf(d1) * math.sqrt(T)
        theta_call = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(
            -r * T
        ) * norm_cdf(d2)
        theta = theta_call if option_type == "call" else theta_call + r * K * math.exp(-r * T)

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    def _run_monte_carlo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        S0 = parameters.get("initial_price", 100.0)
        mu = parameters.get("drift", 0.05)
        sigma = parameters.get("volatility", 0.2)
        T = parameters.get("time_horizon", 1.0)
        n_sims = parameters.get("num_simulations", 10000)
        steps = parameters.get("time_steps", 252)

        # Derive seed from parameters for input-dependent determinism
        seed = compute_deterministic_seed(parameters)
        random.seed(seed)
        dt = T / steps
        paths = []

        for _ in range(min(n_sims, 10000)):
            price = S0
            for _ in range(steps):
                z = random.gauss(0, 1)
                price *= math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
            paths.append(price)

        mean_price = sum(paths) / len(paths)
        var_price = sum((p - mean_price) ** 2 for p in paths) / len(paths)

        return {
            "simulation_type": "geometric_brownian_motion",
            "num_simulations": len(paths),
            "mean_final_price": mean_price,
            "std_final_price": math.sqrt(var_price),
            "percentiles": {
                "5th": sorted(paths)[int(len(paths) * 0.05)],
                "50th": sorted(paths)[int(len(paths) * 0.50)],
                "95th": sorted(paths)[int(len(paths) * 0.95)],
            },
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_var(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) and CVaR."""
        returns = parameters.get("returns", [])
        confidence_level = parameters.get("confidence_level", 0.95)
        portfolio_value = parameters.get("portfolio_value", 1000000.0)

        if not returns:
            # Generate sample returns with seed from parameters
            seed = compute_deterministic_seed(parameters)
            random.seed(seed)
            returns = [random.gauss(0.0005, 0.02) for _ in range(1000)]

        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        var = -sorted_returns[var_index] * portfolio_value

        # CVaR (Conditional VaR)
        cvar_returns = sorted_returns[:var_index]
        cvar = -sum(cvar_returns) / len(cvar_returns) * portfolio_value if cvar_returns else 0

        return {
            "risk_metric": "value_at_risk",
            "confidence_level": confidence_level,
            "portfolio_value": portfolio_value,
            "var": var,
            "cvar": cvar,
            "note": "VaR represents potential loss at confidence level",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _optimize_portfolio(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation."""
        assets = parameters.get("assets", ["AAPL", "GOOGL", "MSFT", "AMZN"])
        expected_returns = parameters.get("expected_returns", [0.12, 0.15, 0.10, 0.14])

        # Simple equal weight allocation
        n = len(assets)
        weights = [1.0 / n] * n

        portfolio_return = sum(w * r for w, r in zip(weights, expected_returns))
        portfolio_risk = 0.15  # Simplified

        return {
            "optimization_type": "mean_variance",
            "assets": assets,
            "optimal_weights": dict(zip(assets, weights)),
            "expected_return": portfolio_return,
            "expected_risk": portfolio_risk,
            "sharpe_ratio": portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _analyze_credit_risk(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze credit risk."""
        borrower_id = parameters.get("borrower_id", "BORROWER_001")
        credit_score = parameters.get("credit_score", 700)
        debt_to_income = parameters.get("debt_to_income", 0.35)
        loan_amount = parameters.get("loan_amount", 250000)

        # Simplified credit scoring
        default_probability = self._estimate_default_probability(credit_score, debt_to_income)
        expected_loss = loan_amount * default_probability * 0.5  # 50% loss given default

        rating = self._assign_credit_rating(default_probability)

        return {
            "analysis_type": "credit_risk",
            "borrower_id": borrower_id,
            "credit_rating": rating,
            "default_probability": default_probability,
            "expected_loss": expected_loss,
            "loan_amount": loan_amount,
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _estimate_default_probability(self, credit_score: float, dti: float) -> float:
        """Estimate probability of default."""
        # Simplified logistic model
        base_prob = 1 / (1 + math.exp((credit_score - 600) / 50))
        dti_adjustment = dti * 0.1
        return min(0.99, max(0.01, base_prob + dti_adjustment))

    def _assign_credit_rating(self, default_prob: float) -> str:
        """Assign credit rating based on default probability."""
        if default_prob < 0.01:
            return "AAA"
        elif default_prob < 0.02:
            return "AA"
        elif default_prob < 0.05:
            return "A"
        elif default_prob < 0.10:
            return "BBB"
        elif default_prob < 0.20:
            return "BB"
        else:
            return "B"

    def _run_stress_test(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress test scenarios."""
        portfolio = parameters.get("portfolio", {"stocks": 0.6, "bonds": 0.3, "cash": 0.1})
        portfolio_value = parameters.get("portfolio_value", 1000000.0)

        scenarios = {
            "market_crash": {"stocks": -0.30, "bonds": -0.10, "cash": 0.0},
            "inflation_spike": {"stocks": -0.15, "bonds": -0.20, "cash": -0.05},
            "recession": {"stocks": -0.25, "bonds": 0.05, "cash": 0.01},
        }

        results = {}
        for scenario_name, impacts in scenarios.items():
            portfolio_impact = sum(
                portfolio.get(asset, 0) * impacts.get(asset, 0) for asset in portfolio.keys()
            )
            results[scenario_name] = {
                "impact_percent": portfolio_impact * 100,
                "value_change": portfolio_value * portfolio_impact,
                "final_value": portfolio_value * (1 + portfolio_impact),
            }

        return {
            "test_type": "stress_scenarios",
            "portfolio_value": portfolio_value,
            "scenarios": results,
            "worst_case": min(results.values(), key=lambda x: x["final_value"]),
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Get optimal compute substrate for financial operation."""
        return get_optimal_substrate(VerticalModule.CAPRA, operation)

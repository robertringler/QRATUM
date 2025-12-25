"""Tests for CAPRA Financial Risk module."""

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  PlatformIntent, VerticalModule)
from verticals.capra import CAPRAModule


class TestCAPRAModule:
    """Test CAPRA Financial Risk module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = CAPRAModule()
        assert module.MODULE_NAME == "CAPRA"
        assert module.MODULE_VERSION == "2.0.0"

    def test_option_pricing(self):
        """Test Black-Scholes option pricing."""
        module = CAPRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.CAPRA,
            operation="option_pricing",
            parameters={
                "spot_price": 100,
                "strike_price": 100,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
            },
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.GB200,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["pricing_model"] == "black_scholes"
        assert "price" in result
        assert "greeks" in result
        assert "delta" in result["greeks"]

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        module = CAPRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.CAPRA,
            operation="monte_carlo",
            parameters={"initial_price": 100, "drift": 0.05, "volatility": 0.2, "num_simulations": 1000},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_002",
            substrate=ComputeSubstrate.GB200,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["simulation_type"] == "geometric_brownian_motion"
        assert "mean_final_price" in result

    def test_var_calculation(self):
        """Test VaR calculation."""
        module = CAPRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.CAPRA,
            operation="var_calculation",
            parameters={"confidence_level": 0.95, "portfolio_value": 1000000},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.GB200,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["risk_metric"] == "value_at_risk"
        assert "var" in result
        assert "cvar" in result

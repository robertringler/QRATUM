"""Tests for FLUXA Supply Chain module."""

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  PlatformIntent, VerticalModule)
from verticals.fluxa import FLUXAModule


class TestFLUXAModule:
    """Test FLUXA Supply Chain module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = FLUXAModule()
        assert module.MODULE_NAME == "FLUXA"
        assert module.MODULE_VERSION == "2.0.0"

    def test_route_optimization(self):
        """Test route optimization."""
        module = FLUXAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.FLUXA,
            operation="route_optimization",
            parameters={
                "depot": {"lat": 0, "lon": 0},
                "locations": [{"id": "A", "lat": 1, "lon": 1, "demand": 10}],
                "num_vehicles": 1,
                "vehicle_capacity": 30,
            },
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.QPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["optimization_type"] == "vehicle_routing"
        assert "routes" in result

    def test_demand_forecasting(self):
        """Test demand forecasting."""
        module = FLUXAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.FLUXA,
            operation="demand_forecasting",
            parameters={"historical_data": [100, 105, 110], "forecast_periods": 3},
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
        assert result["forecast_type"] == "exponential_smoothing"
        assert len(result["forecast"]) == 3

    def test_inventory_optimization(self):
        """Test inventory optimization."""
        module = FLUXAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.FLUXA,
            operation="inventory_optimization",
            parameters={"demand_rate_per_day": 100, "lead_time_days": 7},
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
        assert result["optimization_type"] == "inventory_policy"
        assert "economic_order_quantity" in result

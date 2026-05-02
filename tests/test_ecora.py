"""Tests for ECORA Climate & Energy module."""

from qratum_platform.core import ComputeSubstrate, PlatformContract, PlatformIntent, VerticalModule
from verticals.ecora import ECORAModule


class TestECORAModule:
    """Test ECORA Climate & Energy module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = ECORAModule()
        assert module.MODULE_NAME == "ECORA"
        assert module.MODULE_VERSION == "2.0.0"

    def test_climate_projection(self):
        """Test climate projection with SSP scenarios."""
        module = ECORAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.ECORA,
            operation="climate_projection",
            parameters={"scenario": "SSP2-4.5", "region": "north_america", "target_year": 2100},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_001",
            substrate=ComputeSubstrate.CEREBRAS,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["projection_type"] == "climate_scenario"
        assert "projected_warming_c" in result

    def test_grid_optimization(self):
        """Test energy grid optimization."""
        module = ECORAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.ECORA,
            operation="grid_optimization",
            parameters={"demand_mw": 1000},
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
        assert result["optimization_type"] == "grid_dispatch"
        assert "generation_mix" in result

    def test_carbon_analysis(self):
        """Test carbon footprint analysis."""
        module = ECORAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.ECORA,
            operation="carbon_analysis",
            parameters={"activities": [{"type": "electricity_grid", "amount": 1000}]},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.CPU,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["analysis_type"] == "carbon_footprint"
        assert result["total_emissions_kg_co2e"] > 0

"""Tests for SENTRA Aerospace & Defense module."""

from qratum_platform.core import PlatformContract, PlatformIntent, VerticalModule, ComputeSubstrate
from verticals.sentra import SENTRAModule


class TestSENTRAModule:
    """Test SENTRA Aerospace & Defense module."""

    def test_module_metadata(self):
        """Test module has required metadata."""
        module = SENTRAModule()
        assert module.MODULE_NAME == "SENTRA"
        assert module.MODULE_VERSION == "2.0.0"

    def test_trajectory_simulation(self):
        """Test ballistic trajectory simulation."""
        module = SENTRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.SENTRA,
            operation="trajectory_simulation",
            parameters={"initial_velocity_ms": 500, "launch_angle_deg": 45},
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
        assert result["simulation_type"] == "ballistic_trajectory"
        assert "range_m" in result
        assert "max_height_m" in result

    def test_orbit_propagation(self):
        """Test orbital propagation."""
        module = SENTRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.SENTRA,
            operation="orbit_propagation",
            parameters={"altitude_km": 400, "inclination_deg": 51.6, "duration_hours": 24},
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
        assert result["propagation_type"] == "two_body_keplerian"
        assert "orbital_period_minutes" in result

    def test_radar_analysis(self):
        """Test radar cross-section analysis."""
        module = SENTRAModule()
        intent = PlatformIntent(
            vertical=VerticalModule.SENTRA,
            operation="radar_analysis",
            parameters={"object_type": "sphere", "frequency_ghz": 10, "radius_m": 1},
            user_id="user_001",
        )
        contract = PlatformContract(
            intent=intent,
            contract_id="test_003",
            substrate=ComputeSubstrate.MI300X,
            estimated_cost=0,
            estimated_duration=0,
            safety_checks_passed=True,
        )

        result = module.execute(contract)
        assert result["analysis_type"] == "radar_cross_section"
        assert "rcs_m2" in result

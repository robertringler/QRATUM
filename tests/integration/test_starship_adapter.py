"""Integration tests for Starship vehicle telemetry adapter."""

from __future__ import annotations

import pytest

from telemetry_api import StarshipTelemetryAdapter, StarshipTelemetrySchema


class TestStarshipTelemetryAdapter:
    """Test suite for Starship telemetry adapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes with correct defaults."""
        adapter = StarshipTelemetryAdapter()
        assert adapter.endpoint == "localhost:8003"
        assert not adapter._connected
        assert adapter._schema_version == "1.0"

    def test_adapter_connection(self):
        """Test adapter connection functionality."""
        adapter = StarshipTelemetryAdapter()
        assert adapter.connect()
        assert adapter._connected

        adapter.disconnect()
        assert not adapter._connected

    def test_parse_minimal_telemetry(self):
        """Test parsing minimal valid telemetry data."""
        adapter = StarshipTelemetryAdapter()
        raw_data = {
            "timestamp": 100.5,
            "vehicle_id": "Starship_S25",
            "stage_id": "Ship",
        }

        telemetry = adapter.parse_telemetry(raw_data)

        assert telemetry.timestamp == 100.5
        assert telemetry.vehicle_id == "Starship_S25"
        assert telemetry.stage_id == "Ship"
        assert isinstance(telemetry.dynamics_data, dict)
        assert isinstance(telemetry.propulsion_data, dict)
        assert isinstance(telemetry.gnc_data, dict)
        assert isinstance(telemetry.atmosphere_data, dict)
        assert isinstance(telemetry.reentry_data, dict)

    def test_parse_full_telemetry_booster(self):
        """Test parsing comprehensive booster telemetry data."""
        adapter = StarshipTelemetryAdapter()
        raw_data = {
            "timestamp": 150.0,
            "vehicle_id": "Starship_S25",
            "stage_id": "Booster",
            "position": [0.0, 0.0, 50_000.0],
            "velocity": [100.0, 0.0, 500.0],
            "acceleration": [0.0, 0.0, 20.0],
            "attitude_q": [1.0, 0.0, 0.0, 0.0],
            "angular_vel": [0.0, 0.0, 0.0],
            "angular_accel": [0.0, 0.0, 0.0],
            "mass": 3_700_000.0,  # Total booster mass (dry + propellant)
            "com": [0.0, 0.0, 0.0],
            "raptor_count": 33,
            "raptor_thrust": [2000.0] * 33,
            "raptor_throttle": [100.0] * 33,
            "raptor_gimbal": [0.0] * 33,
            "propellant_mass": 3_400_000.0,  # Propellant mass component
            "propellant_flow": 12_000.0,
            "chamber_pressure": [300.0] * 33,
            "flight_mode": "ASCENT",
            "altitude": 50_000.0,
            "q_dyn": 50_000.0,
            "mach": 2.5,
            "aoa": 0.5,
            "sideslip": 0.0,
            "air_density": 0.5,
        }

        telemetry = adapter.parse_telemetry(raw_data)

        assert telemetry.timestamp == 150.0
        assert telemetry.vehicle_id == "Starship_S25"
        assert telemetry.stage_id == "Booster"
        assert telemetry.dynamics_data["position"] == [0.0, 0.0, 50_000.0]
        assert telemetry.dynamics_data["mass_kg"] == 3_700_000.0
        assert telemetry.propulsion_data["raptor_count"] == 33
        assert telemetry.gnc_data["flight_mode"] == "ASCENT"
        assert telemetry.atmosphere_data["altitude_m"] == 50_000.0
        assert telemetry.atmosphere_data["mach_number"] == 2.5

    def test_parse_full_telemetry_ship_reentry(self):
        """Test parsing comprehensive ship telemetry during reentry."""
        adapter = StarshipTelemetryAdapter()
        raw_data = {
            "timestamp": 500.0,
            "vehicle_id": "Starship_S25",
            "stage_id": "Ship",
            "position": [100_000.0, 0.0, 60_000.0],
            "velocity": [500.0, 0.0, -200.0],
            "acceleration": [0.0, 0.0, -5.0],
            "attitude_q": [0.9239, 0.3827, 0.0, 0.0],
            "angular_vel": [0.1, 0.0, 0.0],
            "angular_accel": [0.0, 0.0, 0.0],
            "mass": 200_000.0,
            "com": [0.0, 0.0, 0.0],
            "raptor_count": 6,
            "raptor_thrust": [0.0] * 6,
            "raptor_throttle": [0.0] * 6,
            "propellant_mass": 800_000.0,
            "flight_mode": "REENTRY",
            "altitude": 60_000.0,
            "q_dyn": 80_000.0,
            "mach": 10.0,
            "aoa": 60.0,
            "sideslip": 0.0,
            "air_density": 0.3,
            "reentry_mode": "BELLY_FLOP",
            "heat_flux": 500.0,
            "stag_temp": 1500.0,
            "grid_fin_deflect": [10.0, -5.0, 0.0, 0.0],
            "landing_legs": "STOWED",
            "descent_mode": "NOMINAL",
        }

        telemetry = adapter.parse_telemetry(raw_data)

        assert telemetry.timestamp == 500.0
        assert telemetry.stage_id == "Ship"
        assert telemetry.gnc_data["flight_mode"] == "REENTRY"
        assert telemetry.atmosphere_data["mach_number"] == 10.0
        assert telemetry.atmosphere_data["angle_of_attack_deg"] == 60.0
        assert telemetry.reentry_data["reentry_mode"] == "BELLY_FLOP"
        assert telemetry.reentry_data["heat_flux_kwpm2"] == 500.0

    def test_parse_missing_required_field(self):
        """Test parsing fails with missing required fields."""
        adapter = StarshipTelemetryAdapter()

        # Missing vehicle_id
        raw_data = {"timestamp": 100.0, "stage_id": "Ship"}
        with pytest.raises(ValueError, match="Missing required field: vehicle_id"):
            adapter.parse_telemetry(raw_data)

        # Missing stage_id
        raw_data = {"timestamp": 100.0, "vehicle_id": "Starship_S25"}
        with pytest.raises(ValueError, match="Missing required field: stage_id"):
            adapter.parse_telemetry(raw_data)

    def test_validate_valid_telemetry(self):
        """Test validation passes for valid telemetry."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            dynamics_data={
                "position": [0.0, 0.0, 1000.0],
                "velocity": [100.0, 0.0, 50.0],
                "attitude_quaternion": [1.0, 0.0, 0.0, 0.0],
                "mass_kg": 200_000.0,
            },
            propulsion_data={
                "raptor_count": 6,
                "propellant_mass_kg": 800_000.0,
            },
            gnc_data={
                "flight_mode": "ASCENT",
            },
            atmosphere_data={
                "altitude_m": 1000.0,
                "mach_number": 0.5,
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert is_valid
        assert len(errors) == 0

    def test_validate_negative_timestamp(self):
        """Test validation fails for negative timestamp."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=-10.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Timestamp must be non-negative" in err for err in errors)

    def test_validate_invalid_vehicle_id(self):
        """Test validation fails for invalid vehicle ID format."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Falcon9_B1067",
            stage_id="Ship",
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Invalid vehicle ID format" in err for err in errors)

    def test_validate_invalid_stage_id(self):
        """Test validation fails for invalid stage ID."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="InvalidStage",
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Invalid stage ID" in err for err in errors)

    def test_validate_unnormalized_quaternion(self):
        """Test validation fails for unnormalized quaternion."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            dynamics_data={
                "attitude_quaternion": [2.0, 0.0, 0.0, 0.0],  # Not normalized
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Quaternion not normalized" in err for err in errors)

    def test_validate_excessive_mass(self):
        """Test validation fails for excessive vehicle mass."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Booster",
            dynamics_data={
                "mass_kg": 10_000_000.0,  # 10,000 tons - too heavy
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Vehicle mass exceeds physical limits" in err for err in errors)

    def test_validate_negative_mass(self):
        """Test validation fails for negative vehicle mass."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            dynamics_data={
                "mass_kg": -100.0,
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Vehicle mass must be non-negative" in err for err in errors)

    def test_validate_invalid_raptor_count(self):
        """Test validation fails for invalid Raptor engine count."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Booster",
            propulsion_data={
                "raptor_count": 100,  # Too many engines
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Invalid Raptor engine count" in err for err in errors)

    def test_validate_invalid_flight_mode(self):
        """Test validation fails for invalid flight mode."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            gnc_data={
                "flight_mode": "INVALID_MODE",
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Invalid flight mode" in err for err in errors)

    def test_validate_altitude_out_of_range(self):
        """Test validation fails for altitude out of range."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            atmosphere_data={
                "altitude_m": 300_000.0,  # Too high
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Altitude out of range" in err for err in errors)

    def test_validate_mach_out_of_range(self):
        """Test validation fails for Mach number out of range."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            atmosphere_data={
                "mach_number": 50.0,  # Too high
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Mach number out of range" in err for err in errors)

    def test_validate_negative_heat_flux(self):
        """Test validation fails for negative heat flux."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=100.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            gnc_data={
                "flight_mode": "REENTRY",
            },
            reentry_data={
                "heat_flux_kwpm2": -100.0,
            },
        )

        is_valid, errors = adapter.validate_schema(telemetry)

        assert not is_valid
        assert any("Heat flux must be non-negative" in err for err in errors)

    def test_ingest_batch_success(self):
        """Test batch ingestion with valid data."""
        adapter = StarshipTelemetryAdapter()
        raw_batch = [
            {
                "timestamp": 100.0,
                "vehicle_id": "Starship_S25",
                "stage_id": "Ship",
                "position": [0.0, 0.0, 1000.0],
                "velocity": [100.0, 0.0, 50.0],
            },
            {
                "timestamp": 101.0,
                "vehicle_id": "Starship_S25",
                "stage_id": "Booster",
                "position": [0.0, 0.0, 2000.0],
                "velocity": [100.0, 0.0, 100.0],
            },
        ]

        successful, failed, errors = adapter.ingest_batch(raw_batch)

        assert successful == 2
        assert failed == 0
        assert len(errors) == 0

    def test_ingest_batch_with_failures(self):
        """Test batch ingestion with some invalid data."""
        adapter = StarshipTelemetryAdapter()
        raw_batch = [
            {
                "timestamp": 100.0,
                "vehicle_id": "Starship_S25",
                "stage_id": "Ship",
            },
            {
                "timestamp": -50.0,  # Invalid negative timestamp
                "vehicle_id": "Starship_S25",
                "stage_id": "Ship",
            },
            {
                "vehicle_id": "Starship_S25",  # Missing timestamp
                "stage_id": "Ship",
            },
        ]

        successful, failed, errors = adapter.ingest_batch(raw_batch)

        assert successful == 1
        assert failed == 2
        assert len(errors) > 0

    def test_export_quasim_format(self):
        """Test export to QuASIM internal format."""
        adapter = StarshipTelemetryAdapter()
        telemetry = StarshipTelemetrySchema(
            timestamp=150.0,
            vehicle_id="Starship_S25",
            stage_id="Ship",
            dynamics_data={
                "position": [0.0, 0.0, 50_000.0],
                "velocity": [100.0, 0.0, 500.0],
                "acceleration": [0.0, 0.0, 20.0],
                "attitude_quaternion": [1.0, 0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "mass_kg": 200_000.0,
            },
            propulsion_data={
                "raptor_count": 6,
                "raptor_thrust_kn": [2000.0] * 6,
                "raptor_throttle_pct": [100.0] * 6,
                "propellant_mass_kg": 800_000.0,
            },
            gnc_data={
                "flight_mode": "ASCENT",
                "guidance_target": {"altitude": 100_000.0},
                "navigation_solution": {"accuracy": 10.0},
                "control_gains": {"kp": 1.0, "ki": 0.1, "kd": 0.01},
            },
            atmosphere_data={
                "altitude_m": 50_000.0,
                "mach_number": 2.5,
            },
            reentry_data={
                "reentry_mode": "NONE",
            },
        )

        quasim_data = adapter.export_quasim_format(telemetry)

        assert quasim_data["timestamp"] == 150.0
        assert quasim_data["source"] == "Starship_Starship_S25_Ship"
        assert quasim_data["dynamics"]["position"] == [0.0, 0.0, 50_000.0]
        assert quasim_data["dynamics"]["velocity"] == [100.0, 0.0, 500.0]
        assert quasim_data["dynamics"]["mass"] == 200_000.0
        assert quasim_data["propulsion"]["engines"] == 6
        assert quasim_data["gnc"]["flight_mode"] == "ASCENT"
        assert "atmosphere" in quasim_data["environment"]
        assert "reentry" in quasim_data["environment"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

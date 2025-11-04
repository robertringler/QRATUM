"""Starship vehicle telemetry adapter for dynamics and GNC scenario data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StarshipTelemetrySchema:
    """Starship vehicle telemetry data schema.

    Attributes:
        timestamp: Mission elapsed time (seconds)
        vehicle_id: Vehicle identifier (e.g., "Starship_S25")
        stage_id: Stage identifier ("Booster" or "Ship")
        dynamics_data: Vehicle dynamics (forces, moments, mass properties)
        propulsion_data: Raptor engine telemetry
        gnc_data: Guidance, navigation, and control data
        atmosphere_data: Atmospheric flight data (when applicable)
        reentry_data: Reentry and landing data (when applicable)
    """

    timestamp: float
    vehicle_id: str
    stage_id: str
    dynamics_data: dict[str, Any] = field(default_factory=dict)
    propulsion_data: dict[str, Any] = field(default_factory=dict)
    gnc_data: dict[str, Any] = field(default_factory=dict)
    atmosphere_data: dict[str, Any] = field(default_factory=dict)
    reentry_data: dict[str, Any] = field(default_factory=dict)


class StarshipTelemetryAdapter:
    """Adapter for Starship vehicle telemetry ingestion."""

    # Stage-specific Raptor engine limits
    MAX_RAPTOR_BOOSTER = 33
    MAX_RAPTOR_SHIP = 6

    def __init__(self, endpoint: str = "localhost:8003"):
        """Initialize Starship telemetry adapter.

        Args:
            endpoint: gRPC endpoint for telemetry service
        """
        self.endpoint = endpoint
        self._connected = False
        self._schema_version = "1.0"

    def connect(self) -> bool:
        """Establish connection to Starship telemetry service.

        Returns:
            True if connection successful
        """
        # In production, would establish actual gRPC connection
        self._connected = True
        return self._connected

    def parse_telemetry(self, raw_data: dict[str, Any]) -> StarshipTelemetrySchema:
        """Parse raw telemetry data into structured schema.

        Args:
            raw_data: Raw telemetry dictionary from Starship data stream

        Returns:
            Parsed telemetry schema

        Raises:
            ValueError: If schema validation fails
        """
        required_fields = ["timestamp", "vehicle_id", "stage_id"]

        for field_name in required_fields:
            if field_name not in raw_data:
                raise ValueError(f"Missing required field: {field_name}")

        # Extract and validate data
        telemetry = StarshipTelemetrySchema(
            timestamp=float(raw_data["timestamp"]),
            vehicle_id=str(raw_data["vehicle_id"]),
            stage_id=str(raw_data["stage_id"]),
            dynamics_data={
                "position": raw_data.get("position", [0.0, 0.0, 0.0]),
                "velocity": raw_data.get("velocity", [0.0, 0.0, 0.0]),
                "acceleration": raw_data.get("acceleration", [0.0, 0.0, 0.0]),
                "attitude_quaternion": raw_data.get("attitude_q", [1.0, 0.0, 0.0, 0.0]),
                "angular_velocity": raw_data.get("angular_vel", [0.0, 0.0, 0.0]),
                "angular_acceleration": raw_data.get("angular_accel", [0.0, 0.0, 0.0]),
                "mass_kg": raw_data.get("mass", 0.0),
                "center_of_mass": raw_data.get("com", [0.0, 0.0, 0.0]),
            },
            propulsion_data={
                "raptor_count": raw_data.get("raptor_count", 0),
                "raptor_thrust_kn": raw_data.get("raptor_thrust", []),
                "raptor_throttle_pct": raw_data.get("raptor_throttle", []),
                "raptor_gimbal_deg": raw_data.get("raptor_gimbal", []),
                "propellant_mass_kg": raw_data.get("propellant_mass", 0.0),
                "propellant_flow_kgps": raw_data.get("propellant_flow", 0.0),
                "chamber_pressure_bar": raw_data.get("chamber_pressure", []),
            },
            gnc_data={
                "flight_mode": raw_data.get("flight_mode", "ASCENT"),
                "guidance_target": raw_data.get("guidance_target", {}),
                "navigation_solution": raw_data.get("nav_solution", {}),
                "control_gains": raw_data.get("control_gains", {}),
                "trajectory_errors": raw_data.get("traj_errors", {}),
                "gps_fix": raw_data.get("gps_fix", False),
                "imu_health": raw_data.get("imu_health", True),
            },
            atmosphere_data={
                "altitude_m": raw_data.get("altitude", 0.0),
                "dynamic_pressure_pa": raw_data.get("q_dyn", 0.0),
                "mach_number": raw_data.get("mach", 0.0),
                "angle_of_attack_deg": raw_data.get("aoa", 0.0),
                "sideslip_angle_deg": raw_data.get("sideslip", 0.0),
                "air_density_kgpm3": raw_data.get("air_density", 0.0),
            },
            reentry_data={
                "reentry_mode": raw_data.get("reentry_mode", "NONE"),
                "heat_flux_kwpm2": raw_data.get("heat_flux", 0.0),
                "stagnation_temp_k": raw_data.get("stag_temp", 0.0),
                "grid_fin_deflection_deg": raw_data.get("grid_fin_deflect", []),
                "landing_leg_status": raw_data.get("landing_legs", "STOWED"),
                "descent_propulsion_mode": raw_data.get("descent_mode", "NOMINAL"),
            },
        )

        return telemetry

    def validate_schema(self, telemetry: StarshipTelemetrySchema) -> tuple[bool, list[str]]:
        """Validate telemetry schema compliance.

        Args:
            telemetry: Parsed telemetry data

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Validate timestamp
        if telemetry.timestamp < 0:
            errors.append("Timestamp must be non-negative")

        # Validate vehicle ID format
        if not telemetry.vehicle_id.startswith("Starship"):
            errors.append(f"Invalid vehicle ID format: {telemetry.vehicle_id}")

        # Validate stage ID
        valid_stages = ["Booster", "Ship"]
        if telemetry.stage_id not in valid_stages:
            errors.append(f"Invalid stage ID: {telemetry.stage_id}")

        # Validate dynamics data
        position = telemetry.dynamics_data.get("position", [0.0, 0.0, 0.0])
        if len(position) != 3:
            errors.append("Position vector must have 3 components")

        velocity = telemetry.dynamics_data.get("velocity", [0.0, 0.0, 0.0])
        if len(velocity) != 3:
            errors.append("Velocity vector must have 3 components")

        # Validate quaternion normalization
        q = telemetry.dynamics_data.get("attitude_quaternion", [1.0, 0.0, 0.0, 0.0])
        if len(q) != 4:
            errors.append("Quaternion must have 4 components")
        else:
            q_norm = sum(qi**2 for qi in q) ** 0.5
            if abs(q_norm - 1.0) > 0.01:
                errors.append(f"Quaternion not normalized: ||q|| = {q_norm}")

        # Validate mass
        mass = telemetry.dynamics_data.get("mass_kg", 0.0)
        if mass < 0:
            errors.append("Vehicle mass must be non-negative")
        elif mass > 5_000_000:  # 5000 tons max
            errors.append(f"Vehicle mass exceeds physical limits: {mass} kg")

        # Validate propulsion data
        raptor_count = telemetry.propulsion_data.get("raptor_count", 0)
        max_raptor = (
            self.MAX_RAPTOR_BOOSTER
            if telemetry.stage_id == "Booster"
            else self.MAX_RAPTOR_SHIP
        )
        if raptor_count < 0 or raptor_count > max_raptor:
            errors.append(f"Invalid Raptor engine count: {raptor_count} (max: {max_raptor})")

        propellant_mass = telemetry.propulsion_data.get("propellant_mass_kg", 0.0)
        if propellant_mass < 0:
            errors.append("Propellant mass must be non-negative")

        # Validate GNC data
        valid_flight_modes = [
            "PRELAUNCH",
            "ASCENT",
            "COAST",
            "REENTRY",
            "DESCENT",
            "LANDING",
            "ABORT",
        ]
        flight_mode = telemetry.gnc_data.get("flight_mode", "ASCENT")
        if flight_mode not in valid_flight_modes:
            errors.append(f"Invalid flight mode: {flight_mode}")

        # Validate atmospheric data
        altitude = telemetry.atmosphere_data.get("altitude_m", 0.0)
        if altitude < -500 or altitude > 200_000:  # Allow below sea level, up to 200km
            errors.append(f"Altitude out of range: {altitude} m")

        mach = telemetry.atmosphere_data.get("mach_number", 0.0)
        if mach < 0 or mach > 30:  # Hypersonic upper limit
            errors.append(f"Mach number out of range: {mach}")

        # Validate reentry data if in reentry mode
        if flight_mode == "REENTRY":
            heat_flux = telemetry.reentry_data.get("heat_flux_kwpm2", 0.0)
            if heat_flux < 0:
                errors.append("Heat flux must be non-negative")

        is_valid = len(errors) == 0
        return is_valid, errors

    def ingest_batch(self, raw_batch: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
        """Ingest batch of telemetry data.

        Args:
            raw_batch: List of raw telemetry dictionaries

        Returns:
            Tuple of (successful_count, failed_count, error_messages)
        """
        successful = 0
        failed = 0
        errors = []

        for i, raw_data in enumerate(raw_batch):
            try:
                telemetry = self.parse_telemetry(raw_data)
                is_valid, validation_errors = self.validate_schema(telemetry)

                if is_valid:
                    successful += 1
                else:
                    failed += 1
                    errors.extend([f"Record {i}: {err}" for err in validation_errors])
            except Exception as e:
                failed += 1
                errors.append(f"Record {i}: {str(e)}")

        return successful, failed, errors

    def export_quasim_format(self, telemetry: StarshipTelemetrySchema) -> dict[str, Any]:
        """Export telemetry in QuASIM internal format.

        Args:
            telemetry: Starship telemetry data

        Returns:
            Dictionary in QuASIM format
        """
        return {
            "timestamp": telemetry.timestamp,
            "source": f"Starship_{telemetry.vehicle_id}_{telemetry.stage_id}",
            "dynamics": {
                "position": telemetry.dynamics_data.get("position", [0.0, 0.0, 0.0]),
                "velocity": telemetry.dynamics_data.get("velocity", [0.0, 0.0, 0.0]),
                "acceleration": telemetry.dynamics_data.get("acceleration", [0.0, 0.0, 0.0]),
                "attitude": telemetry.dynamics_data.get(
                    "attitude_quaternion", [1.0, 0.0, 0.0, 0.0]
                ),
                "angular_velocity": telemetry.dynamics_data.get(
                    "angular_velocity", [0.0, 0.0, 0.0]
                ),
                "mass": telemetry.dynamics_data.get("mass_kg", 0.0),
            },
            "propulsion": {
                "engines": telemetry.propulsion_data.get("raptor_count", 0),
                "thrust": telemetry.propulsion_data.get("raptor_thrust_kn", []),
                "throttle": telemetry.propulsion_data.get("raptor_throttle_pct", []),
                "propellant_mass": telemetry.propulsion_data.get("propellant_mass_kg", 0.0),
            },
            "gnc": {
                "flight_mode": telemetry.gnc_data.get("flight_mode", "ASCENT"),
                "guidance": telemetry.gnc_data.get("guidance_target", {}),
                "navigation": telemetry.gnc_data.get("navigation_solution", {}),
                "control": telemetry.gnc_data.get("control_gains", {}),
            },
            "environment": {
                "atmosphere": telemetry.atmosphere_data,
                "reentry": telemetry.reentry_data,
            },
        }

    def disconnect(self) -> None:
        """Disconnect from telemetry service."""
        self._connected = False

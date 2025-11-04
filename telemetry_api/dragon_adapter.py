"""Dragon spacecraft telemetry adapter for trajectory and GNC data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DragonTelemetrySchema:
    """Dragon spacecraft telemetry data schema.

    Attributes:
        timestamp: Mission elapsed time (seconds)
        vehicle_id: Vehicle identifier (e.g., "Dragon_C210")
        trajectory_data: Trajectory state (position, velocity, acceleration)
        thermal_data: Thermal control system data
        power_data: Power system telemetry
        gnc_data: Guidance, navigation, and control data
        docking_data: Docking system status (when applicable)
    """

    timestamp: float
    vehicle_id: str
    trajectory_data: dict[str, Any] = field(default_factory=dict)
    thermal_data: dict[str, Any] = field(default_factory=dict)
    power_data: dict[str, Any] = field(default_factory=dict)
    gnc_data: dict[str, Any] = field(default_factory=dict)
    docking_data: dict[str, Any] = field(default_factory=dict)


class DragonTelemetryAdapter:
    """Adapter for Dragon spacecraft telemetry ingestion."""

    def __init__(self, endpoint: str = "localhost:8002"):
        """Initialize Dragon telemetry adapter.

        Args:
            endpoint: gRPC endpoint for telemetry service
        """
        self.endpoint = endpoint
        self._connected = False
        self._schema_version = "1.0"

    def connect(self) -> bool:
        """Establish connection to Dragon telemetry service.

        Returns:
            True if connection successful
        """
        # In production, would establish actual gRPC connection
        self._connected = True
        return self._connected

    def parse_telemetry(self, raw_data: dict[str, Any]) -> DragonTelemetrySchema:
        """Parse raw telemetry data into structured schema.

        Args:
            raw_data: Raw telemetry dictionary from Dragon data stream

        Returns:
            Parsed telemetry schema

        Raises:
            ValueError: If schema validation fails
        """
        required_fields = ["timestamp", "vehicle_id"]

        for field_name in required_fields:
            if field_name not in raw_data:
                raise ValueError(f"Missing required field: {field_name}")

        # Extract and validate data
        telemetry = DragonTelemetrySchema(
            timestamp=float(raw_data["timestamp"]),
            vehicle_id=str(raw_data["vehicle_id"]),
            trajectory_data={
                "position_eci": raw_data.get("position", [0.0, 0.0, 0.0]),
                "velocity_eci": raw_data.get("velocity", [0.0, 0.0, 0.0]),
                "acceleration": raw_data.get("acceleration", [0.0, 0.0, 0.0]),
                "altitude_m": raw_data.get("altitude", 0.0),
                "orbital_phase": raw_data.get("orbital_phase", "LEO"),
            },
            thermal_data={
                "radiator_temp_c": raw_data.get("radiator_temp", 0.0),
                "cabin_temp_c": raw_data.get("cabin_temp", 20.0),
                "battery_temp_c": raw_data.get("battery_temp", 20.0),
                "thermal_mode": raw_data.get("thermal_mode", "NOMINAL"),
            },
            power_data={
                "solar_array_voltage_v": raw_data.get("solar_voltage", 0.0),
                "battery_soc_pct": raw_data.get("battery_soc", 100.0),
                "power_draw_w": raw_data.get("power_draw", 0.0),
                "charging_state": raw_data.get("charging", False),
            },
            gnc_data={
                "attitude_quaternion": raw_data.get("attitude_q", [1.0, 0.0, 0.0, 0.0]),
                "angular_velocity": raw_data.get("angular_vel", [0.0, 0.0, 0.0]),
                "control_mode": raw_data.get("control_mode", "NOMINAL"),
                "rcs_propellant_pct": raw_data.get("rcs_propellant", 100.0),
            },
            docking_data={
                "docking_status": raw_data.get("docking_status", "SEPARATED"),
                "relative_range_m": raw_data.get("rel_range", 0.0),
                "relative_velocity_mps": raw_data.get("rel_velocity", 0.0),
                "alignment_error_deg": raw_data.get("align_error", 0.0),
            },
        )

        return telemetry

    def validate_schema(self, telemetry: DragonTelemetrySchema) -> tuple[bool, list[str]]:
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
        if not telemetry.vehicle_id.startswith("Dragon"):
            errors.append(f"Invalid vehicle ID format: {telemetry.vehicle_id}")

        # Validate trajectory data ranges
        position = telemetry.trajectory_data.get("position_eci", [0.0, 0.0, 0.0])
        if len(position) != 3:
            errors.append("Position vector must have 3 components")
        else:
            pos_mag = sum(p**2 for p in position) ** 0.5
            if pos_mag > 0 and (pos_mag < 6_300_000 or pos_mag > 50_000_000):
                errors.append(f"Position magnitude out of orbital range: {pos_mag / 1000:.1f} km")

        velocity = telemetry.trajectory_data.get("velocity_eci", [0.0, 0.0, 0.0])
        if len(velocity) != 3:
            errors.append("Velocity vector must have 3 components")
        else:
            vel_mag = sum(v**2 for v in velocity) ** 0.5
            if vel_mag > 15_000:  # m/s
                errors.append(f"Velocity magnitude exceeds orbital: {vel_mag:.1f} m/s")

        # Validate quaternion normalization
        q = telemetry.gnc_data.get("attitude_quaternion", [1.0, 0.0, 0.0, 0.0])
        if len(q) != 4:
            errors.append("Quaternion must have 4 components")
        else:
            q_norm = sum(qi**2 for qi in q) ** 0.5
            if abs(q_norm - 1.0) > 0.01:
                errors.append(f"Quaternion not normalized: ||q|| = {q_norm}")

        # Validate thermal data
        cabin_temp = telemetry.thermal_data.get("cabin_temp_c", 20.0)
        if cabin_temp < -50 or cabin_temp > 50:
            errors.append(f"Cabin temperature out of safe range: {cabin_temp}°C")

        # Validate power data
        battery_soc = telemetry.power_data.get("battery_soc_pct", 100.0)
        if battery_soc < 0 or battery_soc > 100:
            errors.append(f"Battery SOC out of range: {battery_soc}%")

        # Validate docking data if docking
        docking_status = telemetry.docking_data.get("docking_status", "SEPARATED")
        valid_docking_states = ["SEPARATED", "APPROACH", "PROXIMITY", "DOCKED"]
        if docking_status not in valid_docking_states:
            errors.append(f"Invalid docking status: {docking_status}")

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

    def export_quasim_format(self, telemetry: DragonTelemetrySchema) -> dict[str, Any]:
        """Export telemetry in QuASIM internal format.

        Args:
            telemetry: Dragon telemetry data

        Returns:
            Dictionary in QuASIM format
        """
        return {
            "timestamp": telemetry.timestamp,
            "source": f"Dragon_{telemetry.vehicle_id}",
            "trajectory": {
                "position": telemetry.trajectory_data.get("position_eci", [0.0, 0.0, 0.0]),
                "velocity": telemetry.trajectory_data.get("velocity_eci", [0.0, 0.0, 0.0]),
                "acceleration": telemetry.trajectory_data.get("acceleration", [0.0, 0.0, 0.0]),
                "altitude": telemetry.trajectory_data.get("altitude_m", 0.0),
                "phase": telemetry.trajectory_data.get("orbital_phase", "LEO"),
            },
            "gnc": {
                "attitude": telemetry.gnc_data.get("attitude_quaternion", [1.0, 0.0, 0.0, 0.0]),
                "angular_velocity": telemetry.gnc_data.get("angular_velocity", [0.0, 0.0, 0.0]),
                "mode": telemetry.gnc_data.get("control_mode", "NOMINAL"),
            },
            "systems": {
                "thermal": telemetry.thermal_data,
                "power": telemetry.power_data,
                "docking": telemetry.docking_data,
            },
        }

    def disconnect(self) -> None:
        """Disconnect from telemetry service."""
        self._connected = False

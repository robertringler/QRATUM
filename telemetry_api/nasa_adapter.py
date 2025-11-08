"""NASA telemetry adapter for Orion/SLS GNC data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NASATelemetrySchema:
    """NASA Orion/SLS telemetry data schema.

    Attributes:
        met: Mission Elapsed Time (seconds)
        vehicle_system: Vehicle system identifier ("Orion" or "SLS")
        state_vector: Position and velocity state [x, y, z, vx, vy, vz]
        gnc_mode: GNC operational mode
        control_data: Control system data
        sensor_data: Sensor measurements
    """

    met: float
    vehicle_system: str
    state_vector: list[float] = field(default_factory=list)
    gnc_mode: str = "NOMINAL"
    control_data: dict[str, Any] = field(default_factory=dict)
    sensor_data: dict[str, Any] = field(default_factory=dict)


class NASATelemetryAdapter:
    """Adapter for NASA Orion/SLS telemetry ingestion."""

    def __init__(self, log_format: str = "NASA_CSV_V2"):
        """Initialize NASA telemetry adapter.

        Args:
            log_format: NASA simulation log format version
        """
        self.log_format = log_format
        self._schema_version = "2.0"

    def parse_csv_log(self, csv_line: str) -> NASATelemetrySchema:
        """Parse NASA CSV log line into structured schema.

        Args:
            csv_line: CSV-formatted log line

        Returns:
            Parsed telemetry schema

        Raises:
            ValueError: If parsing fails
        """
        fields = csv_line.strip().split(",")

        if len(fields) < 8:
            raise ValueError(f"Insufficient fields in CSV line: {len(fields)}")

        try:
            telemetry = NASATelemetrySchema(
                met=float(fields[0]),
                vehicle_system=fields[1],
                state_vector=[
                    float(fields[2]),  # x
                    float(fields[3]),  # y
                    float(fields[4]),  # z
                    float(fields[5]),  # vx
                    float(fields[6]),  # vy
                    float(fields[7]),  # vz
                ],
                gnc_mode=fields[8] if len(fields) > 8 else "NOMINAL",
            )

            return telemetry
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse CSV line: {e}") from e

    def parse_json_log(self, json_data: dict[str, Any]) -> NASATelemetrySchema:
        """Parse NASA JSON log entry into structured schema.

        Args:
            json_data: JSON-formatted log entry

        Returns:
            Parsed telemetry schema

        Raises:
            ValueError: If required fields missing
        """
        required_fields = ["MET", "vehicle"]

        for field_name in required_fields:
            if field_name not in json_data:
                raise ValueError(f"Missing required field: {field_name}")

        telemetry = NASATelemetrySchema(
            met=float(json_data["MET"]),
            vehicle_system=json_data["vehicle"],
            state_vector=json_data.get("state", [0.0] * 6),
            gnc_mode=json_data.get("GNC_mode", "NOMINAL"),
            control_data={
                "thruster_commands": json_data.get("thrusters", []),
                "reaction_wheel_torques": json_data.get("rw_torques", [0.0, 0.0, 0.0]),
            },
            sensor_data={
                "star_tracker": json_data.get("star_tracker", {}),
                "imu": json_data.get("imu", {}),
                "gps": json_data.get("gps", {}),
            },
        )

        return telemetry

    def validate_state_vector(self, state_vector: list[float]) -> tuple[bool, str]:
        """Validate state vector dimensions and ranges.

        Args:
            state_vector: [x, y, z, vx, vy, vz] in SI units

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(state_vector) != 6:
            return False, f"State vector must have 6 elements, got {len(state_vector)}"

        # Check position magnitude (Earth orbit: ~6400 km to ~42000 km radius)
        pos_mag = (state_vector[0] ** 2 + state_vector[1] ** 2 + state_vector[2] ** 2) ** 0.5
        if pos_mag < 6_000_000 or pos_mag > 50_000_000:  # meters
            return False, f"Position magnitude out of orbital range: {pos_mag/1000:.1f} km"

        # Check velocity magnitude (orbital: ~7-11 km/s)
        vel_mag = (state_vector[3] ** 2 + state_vector[4] ** 2 + state_vector[5] ** 2) ** 0.5
        if vel_mag > 15_000:  # m/s
            return False, f"Velocity magnitude exceeds orbital: {vel_mag:.1f} m/s"

        return True, ""

    def validate_schema(self, telemetry: NASATelemetrySchema) -> tuple[bool, list[str]]:
        """Validate telemetry schema compliance with NASA standards.

        Args:
            telemetry: Parsed telemetry data

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Validate MET
        if telemetry.met < 0:
            errors.append("MET must be non-negative")

        # Validate vehicle system
        valid_systems = ["Orion", "SLS", "Orion_ESM"]
        if telemetry.vehicle_system not in valid_systems:
            errors.append(f"Invalid vehicle system: {telemetry.vehicle_system}")

        # Validate state vector
        is_valid_state, state_error = self.validate_state_vector(telemetry.state_vector)
        if not is_valid_state:
            errors.append(state_error)

        # Validate GNC mode
        valid_modes = ["NOMINAL", "SAFE", "ABORT", "DOCKING", "REENTRY"]
        if telemetry.gnc_mode not in valid_modes:
            errors.append(f"Invalid GNC mode: {telemetry.gnc_mode}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def ingest_log_file(self, file_path: str) -> tuple[int, int, list[str]]:
        """Ingest NASA simulation log file.

        Args:
            file_path: Path to NASA log file

        Returns:
            Tuple of (successful_count, failed_count, error_messages)
        """
        from pathlib import Path

        file = Path(file_path)
        if not file.exists():
            return 0, 0, [f"File not found: {file_path}"]

        successful = 0
        failed = 0
        errors = []

        try:
            with open(file) as f:
                lines = f.readlines()

            # Skip header if present
            start_idx = 1 if lines and "MET" in lines[0] else 0

            for i, line in enumerate(lines[start_idx:], start=start_idx):
                try:
                    telemetry = self.parse_csv_log(line)
                    is_valid, validation_errors = self.validate_schema(telemetry)

                    if is_valid:
                        successful += 1
                    else:
                        failed += 1
                        errors.extend([f"Line {i}: {err}" for err in validation_errors])
                except Exception as e:
                    failed += 1
                    errors.append(f"Line {i}: {str(e)}")

        except Exception as e:
            return 0, 0, [f"Failed to read file: {str(e)}"]

        return successful, failed, errors

    def export_quasim_format(self, telemetry: NASATelemetrySchema) -> dict[str, Any]:
        """Export telemetry in QuASIM internal format.

        Args:
            telemetry: NASA telemetry data

        Returns:
            Dictionary in QuASIM format
        """
        return {
            "timestamp": telemetry.met,
            "source": "NASA_" + telemetry.vehicle_system,
            "state": {
                "position": telemetry.state_vector[:3],
                "velocity": telemetry.state_vector[3:],
            },
            "mode": telemetry.gnc_mode,
            "control": telemetry.control_data,
            "sensors": telemetry.sensor_data,
        }

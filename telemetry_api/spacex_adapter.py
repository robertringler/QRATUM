"""SpaceX telemetry adapter for Falcon 9 vehicle data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpaceXTelemetrySchema:
    """SpaceX telemetry data schema.

    Attributes:
        timestamp: Mission elapsed time (seconds)
        vehicle_id: Vehicle identifier (e.g., "Falcon9_B1067")
        ascent_data: Ascent trajectory data
        engine_data: Engine telemetry (thrust, ISP, throttle)
        attitude_data: Vehicle attitude (quaternion, rates)
        gnc_loops: GNC feedback control data
    """

    timestamp: float
    vehicle_id: str
    ascent_data: dict[str, Any] = field(default_factory=dict)
    engine_data: dict[str, Any] = field(default_factory=dict)
    attitude_data: dict[str, Any] = field(default_factory=dict)
    gnc_loops: dict[str, Any] = field(default_factory=dict)


class SpaceXTelemetryAdapter:
    """Adapter for SpaceX telemetry ingestion via JSON-RPC/gRPC."""

    def __init__(self, endpoint: str = "localhost:8001"):
        """Initialize SpaceX telemetry adapter.

        Args:
            endpoint: gRPC endpoint for telemetry service
        """
        self.endpoint = endpoint
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to SpaceX telemetry service.

        Returns:
            True if connection successful
        """
        # In production, would establish actual gRPC connection
        self._connected = True
        return self._connected

    def parse_telemetry(self, raw_data: dict[str, Any]) -> SpaceXTelemetrySchema:
        """Parse raw telemetry data into structured schema.

        Args:
            raw_data: Raw telemetry dictionary from SpaceX data stream

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
        telemetry = SpaceXTelemetrySchema(
            timestamp=float(raw_data["timestamp"]),
            vehicle_id=str(raw_data["vehicle_id"]),
            ascent_data={
                "altitude_m": raw_data.get("altitude", 0.0),
                "velocity_mps": raw_data.get("velocity", 0.0),
                "downrange_km": raw_data.get("downrange", 0.0),
            },
            engine_data={
                "thrust_kn": raw_data.get("thrust", 0.0),
                "isp_s": raw_data.get("isp", 0.0),
                "throttle_pct": raw_data.get("throttle", 100.0),
            },
            attitude_data={
                "quaternion": raw_data.get("attitude_q", [1.0, 0.0, 0.0, 0.0]),
                "angular_rates": raw_data.get("angular_rates", [0.0, 0.0, 0.0]),
            },
            gnc_loops={
                "guidance_mode": raw_data.get("guidance_mode", "NOMINAL"),
                "control_gains": raw_data.get("control_gains", {}),
            },
        )

        return telemetry

    def validate_schema(self, telemetry: SpaceXTelemetrySchema) -> tuple[bool, list[str]]:
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
        if not telemetry.vehicle_id.startswith("Falcon"):
            errors.append(f"Invalid vehicle ID format: {telemetry.vehicle_id}")

        # Validate ascent data ranges
        altitude = telemetry.ascent_data.get("altitude_m", 0.0)
        if altitude < 0 or altitude > 500_000:  # 500km upper bound
            errors.append(f"Altitude out of range: {altitude}m")

        # Validate quaternion normalization
        q = telemetry.attitude_data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        if len(q) != 4:
            errors.append("Quaternion must have 4 components")
        else:
            q_norm = sum(qi**2 for qi in q) ** 0.5
            if abs(q_norm - 1.0) > 0.01:
                errors.append(f"Quaternion not normalized: ||q|| = {q_norm}")

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

    def disconnect(self) -> None:
        """Disconnect from telemetry service."""
        self._connected = False

"""Hardware Control & Calibration Layer (HCAL) for QuASIM.

HCAL provides a unified API for hardware control and calibration with:
- Dry-run by default with explicit actuation enablement
- Policy-driven safety enforcement
- Automatic hardware topology discovery
- Real-time telemetry collection
- Closed-loop calibration
- Tamper-evident audit logging
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from quasim.hcal.actuator import Actuator
from quasim.hcal.backends.nvidia_nvml import NvidiaNvmlBackend
from quasim.hcal.loops.calibration import CalibrationResult, bias_trim_v1, power_sweep
from quasim.hcal.policy import PolicyEngine
from quasim.hcal.sensors import SensorManager, TelemetryReading
from quasim.hcal.topology import TopologyDiscovery

__version__ = "0.1.0"


class HCAL:
    """Main HCAL interface for hardware control and calibration."""

    def __init__(
        self,
        policy_path: Optional[Path] = None,
        dry_run: bool = True,
        audit_log_path: Optional[Path] = None,
    ):
        """Initialize HCAL.

        Args:
            policy_path: Path to policy YAML file. If None, uses default policy.
            dry_run: Enable dry-run mode (default: True for safety).
            audit_log_path: Path to audit log file.
        """
        # Initialize policy engine
        self.policy_engine = PolicyEngine(policy_path)

        # Override dry_run if policy requires it
        if self.policy_engine.is_dry_run_default():
            dry_run = True

        self.dry_run = dry_run

        # Initialize components
        self.topology = TopologyDiscovery()
        self.sensor_manager = SensorManager()
        self.actuator = Actuator(
            policy_engine=self.policy_engine,
            audit_log_path=audit_log_path,
            dry_run=dry_run,
        )

        # Initialize backends
        self.backends = {"nvidia_nvml": NvidiaNvmlBackend(dry_run=dry_run)}

        # Discover topology
        self.topology.discover()

    def discover_topology(self):
        """Discover hardware topology.

        Returns:
            Topology instance with discovered hardware.
        """
        return self.topology.discover()

    def apply_setpoint(self, device_id: str, setpoint: Dict[str, Any]) -> bool:
        """Apply setpoint to device.

        Args:
            device_id: Device identifier (e.g., 'gpu0').
            setpoint: Setpoint dictionary with keys like:
                - power_limit_watts: Power cap
                - sm_clock_mhz: SM clock frequency
                - mem_clock_mhz: Memory clock frequency

        Returns:
            True if setpoint was applied successfully.
        """
        # Determine backend
        backend = self._get_backend(device_id)
        if not backend:
            print(f"No backend available for device {device_id}")
            return False

        # Capture baseline before applying
        current_config = backend.read_configuration(device_id)
        self.actuator.capture_baseline(device_id, current_config)

        # Apply setpoint
        return self.actuator.apply_setpoint(device_id, setpoint, backend, validate=True)

    def read_telemetry(self, device_id: str) -> Optional[TelemetryReading]:
        """Read telemetry from device.

        Args:
            device_id: Device identifier.

        Returns:
            TelemetryReading instance or None if failed.
        """
        backend = self._get_backend(device_id)
        if not backend:
            return None

        return self.sensor_manager.read_telemetry(device_id, backend)

    def calibrate_bias_trim(
        self, device_id: str, max_iterations: int = 20
    ) -> CalibrationResult:
        """Run bias trim calibration.

        Args:
            device_id: Device identifier.
            max_iterations: Maximum calibration iterations.

        Returns:
            CalibrationResult instance.
        """
        backend = self._get_backend(device_id)
        if not backend:
            raise ValueError(f"No backend for device {device_id}")

        def measure_fn(dev_id: str, be: Any) -> Dict[str, Any]:
            reading = self.sensor_manager.read_telemetry(dev_id, be)
            return reading.metrics if reading else {}

        def apply_fn(dev_id: str, setpoint: Dict[str, Any], be: Any) -> bool:
            return self.actuator.apply_setpoint(dev_id, setpoint, be, validate=False)

        return bias_trim_v1(device_id, backend, measure_fn, apply_fn)

    def run_power_sweep(
        self, device_id: str, power_range: Tuple[float, float], steps: int = 10
    ):
        """Run power sweep calibration.

        Args:
            device_id: Device identifier.
            power_range: (min_power, max_power) in watts.
            steps: Number of power levels to test.

        Returns:
            List of (power, telemetry) tuples.
        """
        backend = self._get_backend(device_id)
        if not backend:
            raise ValueError(f"No backend for device {device_id}")

        def measure_fn(dev_id: str, be: Any) -> Dict[str, Any]:
            reading = self.sensor_manager.read_telemetry(dev_id, be)
            return reading.metrics if reading else {}

        def apply_fn(dev_id: str, setpoint: Dict[str, Any], be: Any) -> bool:
            return self.actuator.apply_setpoint(dev_id, setpoint, be, validate=False)

        return power_sweep(device_id, backend, measure_fn, apply_fn, power_range, steps)

    def emergency_stop(self):
        """Activate emergency stop - blocks all operations."""
        self.actuator.emergency_stop()

    def get_audit_log(self):
        """Get audit log entries.

        Returns:
            List of audit log entries.
        """
        return self.actuator.get_audit_log()

    def verify_audit_chain(self) -> bool:
        """Verify audit log chain integrity.

        Returns:
            True if chain is valid.
        """
        return self.actuator.verify_audit_chain()

    def _get_backend(self, device_id: str):
        """Get appropriate backend for device.

        Args:
            device_id: Device identifier.

        Returns:
            Backend instance or None.
        """
        # Simple heuristic: gpu* -> nvidia_nvml
        if device_id.startswith("gpu"):
            return self.backends.get("nvidia_nvml")

        return None
"""
Hardware Control Abstraction Layer (HCAL) for QuASIM.

This module provides policy enforcement and safety mechanisms for hardware control operations.
"""

from quasim.hcal.policy import (
    DeviceLimits,
    Environment,
    PolicyEngine,
    PolicyViolation,
)

__all__ = [
    "DeviceLimits",
    "Environment",
    "PolicyEngine",
    "PolicyViolation",
]

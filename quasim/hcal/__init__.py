"""HCAL - Hardware Control Abstraction Layer.

HCAL provides a hardware abstraction layer for controlling and monitoring
physical hardware devices in quantum computing and simulation environments.

Modules:
    device: Device discovery and management
    policy: Policy enforcement and validation
    cli: Command-line interface utilities
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
"""HCAL - Hardware Control Abstraction Layer."""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .actuator import Actuator
from .audit import AuditLogger
from .calibration import CalibrationLoop
from .policy import Policy, PolicyViolation


class HCAL:
    """Hardware Control Abstraction Layer.

    Provides unified interface for hardware discovery, planning,
    actuation, and calibration with policy enforcement and audit logging.
    """

    def __init__(
        self,
        policy: Policy,
        enable_actuation: bool = False,
        audit_logger: Optional[AuditLogger] = None,
"""
Hardware Calibration and Analysis Layer (HCAL) for QuASIM.

This module provides hardware monitoring, calibration, and resource management
capabilities for GPU-accelerated quantum simulation workloads.
"""

__version__ = "0.1.0"
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
from quasim.hcal.loops.calibration import (CalibrationResult, bias_trim_v1,
                                           power_sweep)
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
            policy: Policy instance
            enable_actuation: Whether to enable hardware changes
            audit_logger: Audit logger instance
        """
        self.policy = policy
        self.actuator = Actuator(enable_actuation=enable_actuation)
        self.audit_logger = audit_logger or AuditLogger()

    @classmethod
    def from_policy(
        cls,
        policy_path: Path,
        enable_actuation: bool = False,
        audit_log_dir: Optional[Path] = None,
    ) -> "HCAL":
        """Create HCAL instance from policy file.

        Args:
            policy_path: Path to policy YAML file
            enable_actuation: Whether to enable hardware changes
            audit_log_dir: Directory for audit logs

        Returns:
            HCAL instance
        """
        policy = Policy.from_file(str(policy_path))
        audit_logger = AuditLogger(audit_log_dir) if audit_log_dir else None
        return cls(policy, enable_actuation, audit_logger)

    def discover(self, full: bool = False) -> Dict[str, Any]:
        """Discover hardware topology.

        Args:
            full: Whether to perform full discovery

        Returns:
            Topology dictionary
        """
        topology = {
            "devices": [],
            "interconnects": [],
            "summary": {
                "total_devices": 0,
                "backend_types": list(self.policy.allowed_backends),
            },
        }

        # Mock discovery - in real implementation would query backends
        for backend_type in self.policy.allowed_backends:
            if backend_type == "nvml":
                # Mock NVIDIA GPU discovery
                for device_id in self.policy.device_allowlist:
                    if device_id.startswith("GPU"):
                        topology["devices"].append({
                            "id": device_id,
                            "type": "GPU",
                            "backend": "nvml",
                        })

        topology["summary"]["total_devices"] = len(topology["devices"])

        self.audit_logger.log_event("discover", topology)
        return topology

    def plan(
        self,
        profile: str,
        devices: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create hardware configuration plan.

        Args:
            profile: Configuration profile name
            devices: List of device IDs
            **kwargs: Additional parameters

        Returns:
            Plan dictionary with 'devices' and optional 'warnings' for filtered devices
        """
        plan = {
            "plan_id": str(uuid.uuid4()),
            "profile": profile,
            "devices": {},
        }

        # Generate setpoints based on profile
        if devices is None:
            devices = self.policy.device_allowlist

        filtered_devices = []
        for device_id in devices:
            if device_id in self.policy.device_allowlist:
                # Generate profile-specific setpoints
                setpoints = {}

                if profile == "low-latency":
                    setpoints["power_limit_w"] = 250
                    setpoints["clock_mhz"] = 1800
                elif profile == "balanced":
                    setpoints["power_limit_w"] = 200
                    setpoints["clock_mhz"] = 1500
                elif profile == "power-save":
                    setpoints["power_limit_w"] = 150
                    setpoints["clock_mhz"] = 1200

                plan["devices"][device_id] = setpoints
            else:
                filtered_devices.append(device_id)

        # Add warning if any devices were filtered out
        if filtered_devices:
            plan["warnings"] = {
                "filtered_devices": filtered_devices,
                "reason": "Devices not on allowlist",
            }

        self.audit_logger.log_event("plan", plan)
        return plan

    def apply(
        self,
        plan: Dict[str, Any],
        enable_actuation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Apply a configuration plan.

        Args:
            plan: Plan to apply
            enable_actuation: Override actuation setting

        Returns:
            Application result

        Raises:
            PolicyViolation: If plan violates policy
        """
        # Validate plan against policy
        self.policy.validate_plan(plan)

        # Check if approval required
        if self.policy.requires_approval():
            raise PolicyViolation("Plan requires approval")

        # Apply plan
        if enable_actuation is not None:
            original = self.actuator.enable_actuation
            self.actuator.enable_actuation = enable_actuation
            result = self.actuator.apply_plan(plan)
            self.actuator.enable_actuation = original
        else:
            result = self.actuator.apply_plan(plan)

        self.audit_logger.log_event("apply", result)
        return result

    def calibration(
        self,
        device: str,
        routine: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> CalibrationLoop:
        """Create calibration loop.

        Args:
            device: Device identifier
            routine: Calibration routine name
            parameters: Routine parameters

        Returns:
            CalibrationLoop instance
        """
        if parameters is None:
            parameters = {}

        return CalibrationLoop(
            device=device,
            routine=routine,
            parameters=parameters,
            actuator=self.actuator,
        )

    def get_telemetry(
        self,
        devices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get device telemetry.

        Args:
            devices: List of device IDs

        Returns:
            Telemetry dictionary
        """
        return self.actuator.get_telemetry(devices)

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all operations.

        Returns:
            Stop result
        """
        result = self.actuator.emergency_stop()
        self.audit_logger.log_event("emergency_stop", result)
        return result


__all__ = ["HCAL", "Policy", "PolicyViolation"]
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

from quasim.hcal.policy import (DeviceLimits, Environment, PolicyEngine,
                                PolicyViolation)

__all__ = [
    "DeviceLimits",
    "Environment",
    "PolicyEngine",
    "PolicyViolation",
]

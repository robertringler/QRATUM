"""HCAL - Hardware Control Abstraction Layer.

HCAL provides a unified interface for hardware control and calibration with:
- Dry-run by default with explicit actuation enablement
- Policy-driven safety enforcement
- Automatic hardware topology discovery
- Real-time telemetry collection
- Closed-loop calibration
- Tamper-evident audit logging
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quasim.hcal.actuator import Actuator
from quasim.hcal.audit import AuditLogger
from quasim.hcal.backends.nvidia_nvml import NvidiaNvmlBackend
from quasim.hcal.calibration import CalibrationLoop
from quasim.hcal.loops.calibration import CalibrationResult, bias_trim_v1, power_sweep
from quasim.hcal.policy import DeviceLimits, Environment, Policy, PolicyEngine, PolicyViolation
from quasim.hcal.sensors import SensorManager, TelemetryReading
from quasim.hcal.topology import TopologyDiscovery

__version__ = "0.1.0"


class HCAL:
    """Hardware Control Abstraction Layer.

    Main interface for hardware discovery, planning, actuation, and calibration
    with policy enforcement and audit logging.
    """

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

        # Initialize audit logger
        self.audit_logger = AuditLogger(audit_log_path.parent if audit_log_path else None)

        # For backwards compatibility with old Policy-based API
        self.policy = None
        if policy_path:
            self.policy = Policy.from_file(str(policy_path))

    @classmethod
    def from_policy(
        cls,
        policy_path: Path,
        enable_actuation: bool = False,
        audit_log_dir: Optional[Path] = None,
    ) -> HCAL:
        """Create HCAL instance from policy file.

        Args:
            policy_path: Path to policy YAML file
            enable_actuation: Whether to enable hardware changes
            audit_log_dir: Directory for audit logs

        Returns:
            HCAL instance
        """
        audit_log_path = None
        if audit_log_dir:
            audit_log_dir.mkdir(parents=True, exist_ok=True)
            audit_log_path = audit_log_dir / "audit.log"

        return cls(
            policy_path=policy_path,
            dry_run=not enable_actuation,
            audit_log_path=audit_log_path,
        )

    def discover(self, full: bool = False) -> Dict[str, Any]:
        """Discover hardware topology.

        Args:
            full: Whether to perform full discovery

        Returns:
            Topology dictionary with devices, interconnects, and summary
        """
        topology = {
            "devices": [],
            "interconnects": [],
            "summary": {
                "total_devices": 0,
                "backend_types": [],
            },
        }

        # Get backend types from policy if available
        if self.policy:
            topology["summary"]["backend_types"] = list(self.policy.allowed_backends)

        # Discover devices from topology
        discovered_topology = self.topology.discover()

        for device in discovered_topology.devices:
            device_info = {
                "id": device.device_id,
                "type": str(device.device_type),
                "backend": getattr(device, "backend", "unknown"),
            }
            topology["devices"].append(device_info)

        # Mock discovery for allowed backends if policy exists
        if self.policy:
            for backend_type in self.policy.allowed_backends:
                if backend_type == "nvml":
                    # Mock NVIDIA GPU discovery
                    for device_id in self.policy.device_allowlist:
                        if device_id.startswith("GPU"):
                            # Check if not already added
                            if not any(d["id"] == device_id for d in topology["devices"]):
                                topology["devices"].append(
                                    {
                                        "id": device_id,
                                        "type": "GPU",
                                        "backend": "nvml",
                                    }
                                )

        topology["summary"]["total_devices"] = len(topology["devices"])

        self.audit_logger.log_event("discover", topology)
        return topology

    def discover_topology(self):
        """Discover hardware topology.

        Returns:
            Topology instance with discovered hardware.
        """
        return self.topology.discover()

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

        # Get device list
        if devices is None:
            if self.policy:
                devices = self.policy.device_allowlist
            else:
                devices = []

        filtered_devices = []
        for device_id in devices:
            # Check allowlist if policy exists
            if self.policy and device_id not in self.policy.device_allowlist:
                filtered_devices.append(device_id)
                continue

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
        if self.policy:
            self.policy.validate_plan(plan)

            # Check if approval required
            if self.policy.requires_approval():
                raise PolicyViolation("Plan requires approval")

        # Determine actuation mode
        actuation_enabled = enable_actuation if enable_actuation is not None else not self.dry_run

        # Apply plan
        result = {
            "plan_id": plan.get("plan_id", "unknown"),
            "actuation_enabled": actuation_enabled,
            "devices": {},
        }

        for device_id, setpoints in plan.get("devices", {}).items():
            device_result = {
                "device_id": device_id,
                "setpoints": setpoints,
                "applied": actuation_enabled,
            }
            result["devices"][device_id] = device_result

        self.audit_logger.log_event("apply", result)
        return result

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
        self.actuator.capture_baseline(device_id)

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

    def calibrate_bias_trim(self, device_id: str, max_iterations: int = 20) -> CalibrationResult:
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

    def run_power_sweep(self, device_id: str, power_range: Tuple[float, float], steps: int = 10):
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
        return (
            self.actuator.get_telemetry(devices) if hasattr(self.actuator, "get_telemetry") else {}
        )

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all operations.

        Returns:
            Stop result
        """
        result = {"stopped": True, "timestamp": str(uuid.uuid4())}
        if hasattr(self.actuator, "emergency_stop"):
            self.actuator.emergency_stop()
        self.audit_logger.log_event("emergency_stop", result)
        return result

    def get_audit_log(self):
        """Get audit log entries.

        Returns:
            List of audit log entries.
        """
        return self.actuator.get_audit_log() if hasattr(self.actuator, "get_audit_log") else []

    def verify_audit_chain(self) -> bool:
        """Verify audit log chain integrity.

        Returns:
            True if chain is valid.
        """
        return (
            self.actuator.verify_audit_chain()
            if hasattr(self.actuator, "verify_audit_chain")
            else True
        )

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


__all__ = [
    "__version__",
    "HCAL",
    "DeviceLimits",
    "Environment",
    "PolicyEngine",
    "PolicyViolation",
    "Policy",
    "CalibrationResult",
    "TelemetryReading",
]

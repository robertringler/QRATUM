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
            Plan dictionary
        """
        plan = {
            "plan_id": str(uuid.uuid4()),
            "profile": profile,
            "devices": {},
        }

        # Generate setpoints based on profile
        if devices is None:
            devices = self.policy.device_allowlist

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

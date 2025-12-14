"""HCAL actuator for hardware control."""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditLogEntry:
    """Audit log entry for hardware operations."""

    timestamp: datetime
    operation: str
    device_id: str
    setpoint: Dict[str, Any]
    success: bool
    dry_run: bool
    user: str
    checksum: str
    previous_checksum: Optional[str] = None


@dataclass
class Baseline:
    """Baseline hardware configuration."""

    device_id: str
    timestamp: datetime
    configuration: Dict[str, Any]


class Actuator:
    """Hardware actuator for applying setpoints with policy validation and audit logging."""

    def __init__(
        self,
        enable_actuation: bool = False,
        policy_engine: Optional[Any] = None,
        audit_log_path: Optional[Path] = None,
        dry_run: bool = True,
    ):
        """Initialize actuator.

        Args:
            enable_actuation: Whether to enable actual hardware changes (legacy parameter)
            policy_engine: Policy engine for validation.
            audit_log_path: Path to audit log file.
            dry_run: Enable dry-run mode (default: True).
        """
        # Support both enable_actuation and dry_run parameters
        if enable_actuation is not None:
            self.dry_run = not enable_actuation
            self.enable_actuation = enable_actuation
        else:
            self.dry_run = dry_run
            self.enable_actuation = not dry_run

        self.policy_engine = policy_engine
        self.audit_log_path = audit_log_path or Path("/tmp/hcal_audit.log")
        self._baselines: Dict[str, Baseline] = {}
        self._backends: Dict[str, Any] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._last_checksum: Optional[str] = None
        self._emergency_stop = False

    def capture_baseline(
        self, device_id: str, configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Capture baseline state for a device.

        Args:
            device_id: Device identifier
            configuration: Current device configuration (optional)

        Returns:
            Captured baseline state
        """
        if configuration is None:
            configuration = {}

        baseline_obj = Baseline(
            device_id=device_id,
            timestamp=datetime.now(),
            configuration=configuration.copy(),
        )
        self._baselines[device_id] = baseline_obj

        # Return dict for backwards compatibility
        return {
            "device_id": device_id,
            "timestamp": time.time(),
            "state": configuration,
        }

    def rollback_device(self, device_id: str, backend: Optional[Any] = None) -> Dict[str, Any]:
        """Rollback device to baseline state.

        Args:
            device_id: Device identifier
            backend: Backend driver instance (optional)

        Returns:
            Rollback result
        """
        if device_id not in self._baselines:
            return {
                "success": False,
                "error": f"No baseline captured for {device_id}",
            }

        baseline = self._baselines[device_id]

        # Use provided backend or look up registered one
        if backend is None:
            backend = self._backends.get(device_id)

        # Restore state if backend available and actuation enabled
        if backend and self.enable_actuation:
            try:
                success = backend.apply_setpoint(device_id, baseline.configuration)
                self._log_operation(device_id, "rollback", baseline.configuration, success)
                return {
                    "success": success,
                    "device_id": device_id,
                    "dry_run": False,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        return {
            "success": True,
            "device_id": device_id,
            "dry_run": True,
        }

    def rollback(self, device_id: str, backend: Any) -> bool:
        """Rollback to baseline configuration.

        Args:
            device_id: Device identifier.
            backend: Backend driver instance.

        Returns:
            True if rollback was successful.
        """
        result = self.rollback_device(device_id, backend)
        return result.get("success", False)

    def apply_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a plan to hardware.

        Args:
            plan: Plan to apply

        Returns:
            Application result
        """
        result = {
            "plan_id": plan.get("plan_id", "unknown"),
            "actuation_enabled": self.enable_actuation,
            "devices": {},
        }

        devices = plan.get("devices", {})
        for device_id, setpoints in devices.items():
            backend = self._backends.get(device_id)

            if backend and self.enable_actuation:
                device_result = backend.apply_setpoint(device_id, setpoints)
            else:
                device_result = {
                    "success": True,
                    "dry_run": True,
                    "setpoints": setpoints,
                }

            result["devices"][device_id] = device_result

        return result

    def apply_setpoint(
        self,
        device_id: str,
        setpoint: Dict[str, Any],
        backend: Any,
        validate: bool = True,
    ) -> bool:
        """Apply setpoint to device.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary.
            backend: Backend driver instance.
            validate: Enable pre/post validation.

        Returns:
            True if setpoint was applied successfully.
        """
        if self._emergency_stop:
            print("Emergency stop active - operation blocked")
            return False

        # Policy engine checks if available
        if self.policy_engine:
            # Check rate limit
            if hasattr(self.policy_engine, "check_rate_limit"):
                if not self.policy_engine.check_rate_limit():
                    print("Rate limit exceeded")
                    return False

            # Check device allowed
            if hasattr(self.policy_engine, "check_device_allowed"):
                if not self.policy_engine.check_device_allowed(device_id):
                    print(f"Device {device_id} not allowed by policy")
                    return False

            # Check backend allowed
            if hasattr(self.policy_engine, "check_backend_allowed"):
                backend_name = backend.__class__.__name__
                if not self.policy_engine.check_backend_allowed(backend_name):
                    print(f"Backend {backend_name} not allowed by policy")
                    return False

            # Check limits
            if hasattr(self.policy_engine, "check_limits"):
                if not self.policy_engine.check_limits(device_id, setpoint):
                    print(f"Setpoint exceeds limits for device {device_id}")
                    return False

        # Pre-validation
        if validate and not self._pre_validate(device_id, setpoint, backend):
            print("Pre-validation failed")
            return False

        # Apply setpoint (or simulate in dry-run)
        success = False
        try:
            if self.dry_run:
                print(f"[DRY-RUN] Would apply setpoint to {device_id}: {setpoint}")
                success = True
            else:
                success = backend.apply_setpoint(device_id, setpoint)

            # Post-validation
            if validate and not self.dry_run:
                if not self._post_validate(device_id, setpoint, backend):
                    print("Post-validation failed - rolling back")
                    self.rollback(device_id, backend)
                    success = False

        except Exception as e:
            print(f"Error applying setpoint: {e}")
            if not self.dry_run:
                self.rollback(device_id, backend)
            success = False

        # Log operation
        self._log_operation(device_id, "apply_setpoint", setpoint, success)

        return success

    def _pre_validate(self, device_id: str, setpoint: Dict[str, Any], backend: Any) -> bool:
        """Pre-validate setpoint before application.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary.
            backend: Backend driver instance.

        Returns:
            True if validation passes.
        """
        # Check if device exists
        if hasattr(backend, "device_exists") and not backend.device_exists(device_id):
            return False

        # Check if setpoint is valid
        return setpoint

    def _post_validate(self, device_id: str, setpoint: Dict[str, Any], backend: Any) -> bool:
        """Post-validate setpoint after application.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary.
            backend: Backend driver instance.

        Returns:
            True if validation passes.
        """
        # Read back configuration
        try:
            if not hasattr(backend, "read_configuration"):
                return True

            actual = backend.read_configuration(device_id)

            # Check if setpoint was applied (with tolerance)
            for key, expected_value in setpoint.items():
                if key not in actual:
                    return False

                actual_value = actual[key]

                # Allow 5% tolerance for numeric values
                if isinstance(expected_value, (int, float)):
                    tolerance = abs(expected_value * 0.05)
                    if abs(actual_value - expected_value) > tolerance:
                        return False
                elif actual_value != expected_value:
                    return False

            return True

        except Exception as e:
            print(f"Post-validation error: {e}")
            return False

    def register_backend(self, device_id: str, backend: Any) -> None:
        """Register a backend for a device.

        Args:
            device_id: Device identifier
            backend: Backend instance
        """
        self._backends[device_id] = backend

    def get_telemetry(self, device_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get telemetry from devices.

        Args:
            device_ids: List of device IDs, or None for all

        Returns:
            Telemetry data
        """
        telemetry = {}

        if device_ids is None:
            device_ids = list(self._backends.keys())

        for device_id in device_ids:
            backend = self._backends.get(device_id)
            if backend and hasattr(backend, "get_telemetry"):
                telemetry[device_id] = backend.get_telemetry()

        return telemetry

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all devices.

        Returns:
            Stop result
        """
        print("EMERGENCY STOP ACTIVATED")
        self._emergency_stop = True
        self._log_operation("system", "emergency_stop", {}, True)

        return {
            "timestamp": time.time(),
            "devices_stopped": list(self._backends.keys()),
        }

    def reset_emergency_stop(self):
        """Reset emergency stop."""
        print("Emergency stop reset")
        self._emergency_stop = False

    def _log_operation(
        self,
        device_id: str,
        operation: str,
        setpoint: Dict[str, Any],
        success: bool,
    ):
        """Log operation to audit log.

        Args:
            device_id: Device identifier.
            operation: Operation name.
            setpoint: Setpoint dictionary.
            success: Operation success status.
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            operation=operation,
            device_id=device_id,
            setpoint=setpoint,
            success=success,
            dry_run=self.dry_run,
            user="system",  # In production, use actual user
            checksum="",
            previous_checksum=self._last_checksum,
        )

        # Compute tamper-evident checksum
        entry_data = {
            "timestamp": entry.timestamp.isoformat(),
            "operation": entry.operation,
            "device_id": entry.device_id,
            "setpoint": entry.setpoint,
            "success": entry.success,
            "dry_run": entry.dry_run,
            "user": entry.user,
            "previous_checksum": entry.previous_checksum,
        }
        entry_json = json.dumps(entry_data, sort_keys=True)
        entry.checksum = hashlib.sha256(entry_json.encode()).hexdigest()
        self._last_checksum = entry.checksum

        self._audit_log.append(entry)

        # Write to file
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry_data) + "\n")
        except Exception as e:
            print(f"Error writing audit log: {e}")

    def get_audit_log(self) -> List[AuditLogEntry]:
        """Get audit log entries.

        Returns:
            List of audit log entries.
        """
        return self._audit_log.copy()

    def verify_audit_chain(self) -> bool:
        """Verify audit log chain integrity.

        Returns:
            True if chain is valid.
        """
        previous_checksum = None

        for entry in self._audit_log:
            if entry.previous_checksum != previous_checksum:
                return False

            # Recompute checksum
            entry_data = {
                "timestamp": entry.timestamp.isoformat(),
                "operation": entry.operation,
                "device_id": entry.device_id,
                "setpoint": entry.setpoint,
                "success": entry.success,
                "dry_run": entry.dry_run,
                "user": entry.user,
                "previous_checksum": entry.previous_checksum,
            }
            entry_json = json.dumps(entry_data, sort_keys=True)
            expected_checksum = hashlib.sha256(entry_json.encode()).hexdigest()

            if entry.checksum != expected_checksum:
                return False

            previous_checksum = entry.checksum

        return True


__all__ = ["Actuator", "AuditLogEntry", "Baseline"]

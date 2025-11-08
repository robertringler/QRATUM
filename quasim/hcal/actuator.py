"""Hardware actuator for HCAL - unified setpoint application."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from quasim.hcal.policy import PolicyEngine


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
    """Hardware actuator for applying setpoints."""

    def __init__(
        self,
        policy_engine: PolicyEngine,
        audit_log_path: Optional[Path] = None,
        dry_run: bool = True,
    ):
        """Initialize actuator.

        Args:
            policy_engine: Policy engine for validation.
            audit_log_path: Path to audit log file.
            dry_run: Enable dry-run mode (default: True).
        """
        self.policy_engine = policy_engine
        self.audit_log_path = audit_log_path or Path("/tmp/hcal_audit.log")
        self.dry_run = dry_run
        self._baselines: Dict[str, Baseline] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._last_checksum: Optional[str] = None
        self._emergency_stop = False

    def capture_baseline(self, device_id: str, configuration: Dict[str, Any]):
        """Capture baseline configuration for rollback.

        Args:
            device_id: Device identifier.
            configuration: Current device configuration.
        """
        baseline = Baseline(
            device_id=device_id, timestamp=datetime.now(), configuration=configuration.copy()
        )
        self._baselines[device_id] = baseline

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

        # Check rate limit
        if not self.policy_engine.check_rate_limit():
            print("Rate limit exceeded")
            return False

        # Check device allowed
        if not self.policy_engine.check_device_allowed(device_id):
            print(f"Device {device_id} not allowed by policy")
            return False

        # Check backend allowed
        backend_name = backend.__class__.__name__
        if not self.policy_engine.check_backend_allowed(backend_name):
            print(f"Backend {backend_name} not allowed by policy")
            return False

        # Check limits
        if not self.policy_engine.check_limits(device_id, setpoint):
            print(f"Setpoint exceeds limits for device {device_id}")
            return False

        # Pre-validation
        if validate:
            if not self._pre_validate(device_id, setpoint, backend):
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
        if not backend.device_exists(device_id):
            return False

        # Check if setpoint is valid
        if not setpoint:
            return False

        return True

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

    def rollback(self, device_id: str, backend: Any) -> bool:
        """Rollback to baseline configuration.

        Args:
            device_id: Device identifier.
            backend: Backend driver instance.

        Returns:
            True if rollback was successful.
        """
        if device_id not in self._baselines:
            print(f"No baseline for device {device_id}")
            return False

        baseline = self._baselines[device_id]
        print(f"Rolling back {device_id} to baseline from {baseline.timestamp}")

        try:
            success = backend.apply_setpoint(device_id, baseline.configuration)
            self._log_operation(device_id, "rollback", baseline.configuration, success)
            return success
        except Exception as e:
            print(f"Rollback error: {e}")
            return False

    def emergency_stop(self):
        """Activate emergency stop - blocks all operations."""
        print("EMERGENCY STOP ACTIVATED")
        self._emergency_stop = True
        self._log_operation("system", "emergency_stop", {}, True)

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

"""
Policy engine for hardware control operations.

Provides validation, rate limiting, and approval mechanisms for safe hardware control.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


class Environment(Enum):
    """Environment types for policy enforcement."""

    DEV = "DEV"
    LAB = "LAB"
    PROD = "PROD"


class PolicyViolation(Exception):
    """Exception raised when a policy is violated."""

    pass


@dataclass
class DeviceLimits:
    """Device limits for hardware control operations."""

    power_watts_max: Optional[float] = None
    temp_c_max: Optional[float] = None
    voltage_mv_range: Optional[Tuple[float, float]] = None
    freq_mhz_range: Optional[Tuple[float, float]] = None

    def validate_setpoint(self, parameter: str, value: float) -> None:
        """
        Validate a setpoint against device limits.

        Args:
            parameter: Parameter name (e.g., 'power_watts', 'temp_c')
            value: Value to validate

        Raises:
            PolicyViolation: If the setpoint violates limits
        """
        if parameter == "power_watts":
            if self.power_watts_max is not None and value > self.power_watts_max:
                raise PolicyViolation(
                    f"Power setpoint {value}W exceeds maximum {self.power_watts_max}W"
                )
        elif parameter == "temp_c":
            if self.temp_c_max is not None and value > self.temp_c_max:
                raise PolicyViolation(f"Temperature {value}°C exceeds maximum {self.temp_c_max}°C")
        elif parameter == "voltage_mv":
            if self.voltage_mv_range is not None:
                min_v, max_v = self.voltage_mv_range
                if value < min_v or value > max_v:
                    raise PolicyViolation(f"Voltage {value}mV outside range [{min_v}, {max_v}]mV")
        elif parameter == "freq_mhz" and self.freq_mhz_range is not None:
            min_f, max_f = self.freq_mhz_range
            if value < min_f or value > max_f:
                raise PolicyViolation(f"Frequency {value}MHz outside range [{min_f}, {max_f}]MHz")


@dataclass
class RateLimiter:
    """Rate limiter for hardware control operations."""

    commands_per_minute: int = 60
    window_seconds: int = 60
    _timestamps: deque = field(default_factory=deque, init=False, repr=False)

    def check_and_record(self) -> None:
        """
        Check rate limit and record the current command.

        Raises:
            PolicyViolation: If rate limit is exceeded
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old timestamps outside the window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        # Check if we've exceeded the limit
        if len(self._timestamps) >= self.commands_per_minute:
            raise PolicyViolation(
                f"Rate limit exceeded: {self.commands_per_minute} commands per {self.window_seconds} seconds"
            )

        # Record the current timestamp
        self._timestamps.append(now)


class PolicyEngine:
    """
    Policy engine for hardware control operations.

    Provides validation, rate limiting, and approval mechanisms.
    """

    def __init__(self, policy_file: Optional[Path] = None):
        """
        Initialize policy engine.

        Args:
            policy_file: Path to policy configuration YAML file
        """
        # Default configuration
        self.config = {
            "environment": "DEV",
            "allowed_backends": ["nvml", "rocm_smi"],
            "device_allowlist": None,  # None means all devices allowed
            "limits": {
                "power_watts_max": 350,
                "temp_c_max": 90,
            },
            "rate_limits": {
                "commands_per_minute": 60,
            },
            "approvals": {
                "required": True,
                "method": "file_token",
                "token_path": None,
            },
        }

        # Load from file if provided
        if policy_file is not None:
            with open(policy_file) as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)

        # Set environment
        env_str = self.config.get("environment", "DEV")
        self.environment = Environment[env_str]

        # Create device limits
        limits_config = self.config.get("limits", {})
        self.limits = DeviceLimits(
            power_watts_max=limits_config.get("power_watts_max"),
            temp_c_max=limits_config.get("temp_c_max"),
            voltage_mv_range=limits_config.get("voltage_mv_range"),
            freq_mhz_range=limits_config.get("freq_mhz_range"),
        )

        # Create rate limiter
        rate_limits = self.config.get("rate_limits", {})
        self.rate_limiter = RateLimiter(
            commands_per_minute=rate_limits.get("commands_per_minute", 60),
            window_seconds=rate_limits.get("window_seconds", 60),
        )

    def requires_approval(self) -> bool:
        """
        Check if approval is required for operations.

        Returns:
            True if approval is required
        """
        return self.config.get("approvals", {}).get("required", False)

    def is_backend_allowed(self, backend: str) -> bool:
        """
        Check if a backend is allowed.

        Args:
            backend: Backend name

        Returns:
            True if backend is allowed
        """
        allowed = self.config.get("allowed_backends", [])
        return backend in allowed

    def is_device_allowed(self, device_id: str) -> bool:
        """
        Check if a device is allowed.

        Args:
            device_id: Device identifier

        Returns:
            True if device is allowed
        """
        allowlist = self.config.get("device_allowlist")
        if allowlist is None:
            return True
        return device_id in allowlist

    def validate_operation(
        self,
        device_id: str,
        operation: str,
        setpoints: Dict[str, Any],
        enable_actuation: bool,
    ) -> None:
        """
        Validate a hardware control operation.

        Args:
            device_id: Device identifier
            operation: Operation name
            setpoints: Setpoint parameters
            enable_actuation: Whether actuation is enabled

        Raises:
            PolicyViolation: If operation violates policy
        """
        # Check device allowlist
        if not self.is_device_allowed(device_id):
            raise PolicyViolation(f"Device {device_id} not in allowlist")

        # Check production environment restrictions
        if self.environment == Environment.PROD and operation == "firmware_update":
            raise PolicyViolation("Firmware updates not allowed in PROD environment")

        # Validate setpoints against limits
        for param, value in setpoints.items():
            # Map setpoint parameters to validation parameters
            if param == "power_limit_w":
                self.limits.validate_setpoint("power_watts", value)
            elif param.startswith("temp"):
                self.limits.validate_setpoint("temp_c", value)
            elif param.startswith("voltage"):
                self.limits.validate_setpoint("voltage_mv", value)
            elif param.startswith("freq"):
                self.limits.validate_setpoint("freq_mhz", value)

    def validate_approval(self, token: str) -> None:
        """
        Validate an approval token.

        Args:
            token: Approval token to validate

        Raises:
            PolicyViolation: If token is invalid
        """
        approvals = self.config.get("approvals", {})
        method = approvals.get("method")

        if method == "file_token":
            token_path = approvals.get("token_path")
            if token_path is None:
                raise PolicyViolation("Token path not configured")

            token_file = Path(token_path)
            if not token_file.exists():
                raise PolicyViolation(f"Token file not found: {token_path}")

            expected_token = token_file.read_text().strip()
            if token != expected_token:
                raise PolicyViolation("Invalid approval token")

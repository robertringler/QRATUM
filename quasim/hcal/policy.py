"""Policy enforcement and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class PolicyConfig:
    """Policy configuration.

    Attributes:
        environment: Environment type (DEV, LAB, PROD)
        allowed_backends: List of allowed compute backends
        limits: Resource limits
        raw_config: Raw configuration dictionary
    """

    environment: str
    allowed_backends: list[str]
    limits: dict[str, Any]
    raw_config: dict[str, Any] = field(default_factory=dict)


class PolicyValidator:
    """Validates and enforces hardware control policies."""

    def __init__(self, policy_config: PolicyConfig | None = None) -> None:
        """Initialize policy validator.

        Args:
            policy_config: Policy configuration
        """

        self.policy = policy_config

    @classmethod
    def from_file(cls, policy_path: Path) -> PolicyValidator:
        """Load policy from YAML file.

        Args:
            policy_path: Path to policy YAML file

        Returns:
            PolicyValidator instance

        Raises:
            FileNotFoundError: If policy file doesn't exist
            ValueError: If policy is invalid
        """

        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        with open(policy_path) as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Policy file is empty or invalid: {policy_path}")

        cls._validate_config(config)

        policy = PolicyConfig(
            environment=config["environment"],
            allowed_backends=config["allowed_backends"],
            limits=config["limits"],
            raw_config=config,
        )

        return cls(policy)

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        """Validate policy configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """

        required_keys = ["environment", "allowed_backends", "limits"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required policy key: {key}")

        valid_environments = ["DEV", "LAB", "PROD"]
        if config["environment"] not in valid_environments:
            raise ValueError(
                f"Invalid environment: {config['environment']}. Must be one of {valid_environments}"
            )

    def validate_backend(self, backend: str) -> bool:
        """Validate if backend is allowed.

        Args:
            backend: Backend name to validate

        Returns:
            True if backend is allowed
        """

        if not self.policy:
            return True

        return backend in self.policy.allowed_backends

    def check_limits(self, resource: str, value: int) -> bool:
        """Check if resource value is within limits.

        Args:
            resource: Resource name
            value: Resource value to check

        Returns:
            True if within limits
        """

        if not self.policy or resource not in self.policy.limits:
            return True

        try:
            limit = int(self.policy.limits[resource])
        except (ValueError, TypeError):
            # Invalid limit value; treat as policy violation
            return False
        return value <= limit


"""HCAL policy enforcement and validation."""


"""Policy engine for HCAL - declarative YAML-based policy configuration."""

import time
from dataclasses import dataclass, field
from enum import Enum

"""

Policy engine for hardware control operations.

Provides validation, rate limiting, and approval mechanisms for safe hardware control.
"""

from collections import deque
from dataclasses import dataclass, field


class PolicyViolation(Exception):
    """Raised when a policy is violated."""

    pass


class Policy:
    """Hardware control policy."""

    def __init__(self, config: dict[str, Any]):
        """Initialize policy from configuration.

        Args:
            config: Policy configuration dictionary
        """

        self.config = config
        self.environment = config.get("environment", "DEV")
        self.allowed_backends = config.get("allowed_backends", [])
        self.device_allowlist = config.get("device_allowlist", [])
        self.limits = config.get("limits", {})
        self.rate_limits = config.get("rate_limits", {})
        self.approvals = config.get("approvals", {})

    @classmethod
    def from_file(cls, path: str) -> Policy:
        """Load policy from YAML file.

        Args:
            path: Path to policy YAML file

        Returns:
            Policy instance
        """

        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def validate_plan(self, plan: dict[str, Any]) -> None:
        """Validate a plan against policy.

        Args:
            plan: Plan dictionary

        Raises:
            PolicyViolation: If plan violates policy
        """

        devices = plan.get("devices", {})

        for device_id, setpoints in devices.items():
            # Check device allowlist
            if self.device_allowlist and device_id not in self.device_allowlist:
                raise PolicyViolation(f"Device {device_id} not on allowlist")

            # Check power limits
            if "power_limit_w" in setpoints:
                power = setpoints["power_limit_w"]
                max_power = self.limits.get("power_watts_max")
                if max_power and power > max_power:
                    raise PolicyViolation(f"Power limit {power}W exceeds maximum {max_power}W")

            # Check temperature limits
            if "temp_limit_c" in setpoints:
                temp = setpoints["temp_limit_c"]
                max_temp = self.limits.get("temp_c_max")
                if max_temp and temp > max_temp:
                    raise PolicyViolation(
                        f"Temperature limit {temp}°C exceeds maximum {max_temp}°C"
                    )

    def requires_approval(self) -> bool:
        """Check if approvals are required.

        Returns:
            True if approvals required
        """

        return self.approvals.get("required", False)


class Environment(Enum):
    """Environment types for policy enforcement."""

    DEV = "dev"
    LAB = "lab"
    PROD = "prod"


class PolicyViolationError(Exception):
    """Exception raised when a policy is violated."""

    pass


# Backwards compatibility alias
PolicyViolation = PolicyViolationError


@dataclass
class DeviceLimits:
    """Device limits for hardware control operations."""

    power_watts_max: float | None = None
    temp_c_max: float | None = None
    voltage_mv_range: tuple[float, float] | None = None
    freq_mhz_range: tuple[float, float] | None = None

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

    def __init__(self, policy_file: Path | None = None):
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

        # Set environment. Accept either the enum name ("DEV") or the enum
        # value ("dev"), since policy YAML files use the lowercase value form.
        env_str = str(self.config.get("environment", "DEV"))
        self.environment = Environment[env_str.upper()]

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
        setpoints: dict[str, Any],
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

        # Check rate limit first (before any other validation)
        try:
            self.rate_limiter.check_and_record()
        except PolicyViolation:
            # Re-raise with more context
            raise PolicyViolation("Rate limit exceeded - operation blocked")

        # Check device allowlist
        if not self.is_device_allowed(device_id):
            raise PolicyViolation(f"Device {device_id} not in allowlist")

        # Check production environment restrictions (before approval check)
        if self.environment == Environment.PROD and operation == "firmware_update":
            raise PolicyViolation("Firmware updates not allowed in PROD environment")

        # Check if approval is required for actuation operations
        if enable_actuation and self.requires_approval():
            raise PolicyViolation(
                "Operation requires approval but no approval provided. "
                "Call validate_approval() with a valid token before actuating."
            )

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

    def is_dry_run_default(self) -> bool:
        """

        Check if dry-run mode is the default for this policy.

        Returns:
            True if dry-run is the default mode
        """

        return self.config.get("dry_run_default", True)

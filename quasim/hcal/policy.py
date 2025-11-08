"""Policy engine for HCAL - declarative YAML-based policy configuration."""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
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

    DEV = "dev"
    LAB = "lab"
    PROD = "prod"


@dataclass
class DeviceLimits:
    """Hardware limits for devices."""

    max_power_watts: Optional[float] = None
    max_temp_celsius: Optional[float] = None
    max_voltage_volts: Optional[float] = None
    max_clock_mhz: Optional[float] = None
    min_power_watts: Optional[float] = None
    min_temp_celsius: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    commands_per_minute: int = 30
    window_seconds: int = 60


@dataclass
class ApprovalGate:
    """Approval gate configuration."""

    required: bool = False
    min_approvers: int = 0
    methods: List[str] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Complete policy configuration."""

    environment: Environment
    device_allowlist: List[str]
    backend_restrictions: List[str]
    device_limits: Dict[str, DeviceLimits]
    rate_limit: RateLimitConfig
    approval_gate: ApprovalGate
    dry_run_default: bool = True


class PolicyEngine:
    """Policy engine for HCAL operations."""

    def __init__(self, policy_path: Optional[Path] = None):
        """Initialize policy engine.

        Args:
            policy_path: Path to policy YAML file. If None, uses default policy.
        """
        self.policy_path = policy_path
        self.policy: Optional[PolicyConfig] = None
        self._command_history: List[float] = []

        if policy_path:
            self.load_policy(policy_path)
        else:
            self._load_default_policy()

    def _load_default_policy(self):
        """Load default restrictive policy."""
        self.policy = PolicyConfig(
            environment=Environment.DEV,
            device_allowlist=["*"],
            backend_restrictions=[],
            device_limits={},
            rate_limit=RateLimitConfig(),
            approval_gate=ApprovalGate(),
            dry_run_default=True,
        )

    def load_policy(self, policy_path: Path):
        """Load policy from YAML file.

        Args:
            policy_path: Path to policy YAML file.

        Raises:
            ValueError: If policy is invalid.
        """
        with open(policy_path) as f:
            data = yaml.safe_load(f)

        self._validate_schema(data)
        self.policy = self._parse_policy(data)

    def _validate_schema(self, data: Dict[str, Any]):
        """Validate policy schema.

        Args:
            data: Policy data dictionary.

        Raises:
            ValueError: If schema is invalid.
        """
        required_fields = ["environment", "device_allowlist", "backend_restrictions"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        if data["environment"] not in ["dev", "lab", "prod"]:
            raise ValueError(f"Invalid environment: {data['environment']}")

    def _parse_policy(self, data: Dict[str, Any]) -> PolicyConfig:
        """Parse policy data into PolicyConfig.

        Args:
            data: Policy data dictionary.

        Returns:
            PolicyConfig instance.
        """
        # Parse device limits
        device_limits = {}
        for device, limits in data.get("device_limits", {}).items():
            device_limits[device] = DeviceLimits(
                max_power_watts=limits.get("max_power_watts"),
                max_temp_celsius=limits.get("max_temp_celsius"),
                max_voltage_volts=limits.get("max_voltage_volts"),
                max_clock_mhz=limits.get("max_clock_mhz"),
                min_power_watts=limits.get("min_power_watts"),
                min_temp_celsius=limits.get("min_temp_celsius"),
            )

        # Parse rate limit
        rate_limit_data = data.get("rate_limit", {})
        rate_limit = RateLimitConfig(
            commands_per_minute=rate_limit_data.get("commands_per_minute", 30),
            window_seconds=rate_limit_data.get("window_seconds", 60),
        )

        # Parse approval gate
        approval_data = data.get("approval_gate", {})
        approval_gate = ApprovalGate(
            required=approval_data.get("required", False),
            min_approvers=approval_data.get("min_approvers", 0),
            methods=approval_data.get("methods", []),
        )

        return PolicyConfig(
            environment=Environment(data["environment"]),
            device_allowlist=data["device_allowlist"],
            backend_restrictions=data["backend_restrictions"],
            device_limits=device_limits,
            rate_limit=rate_limit,
            approval_gate=approval_gate,
            dry_run_default=data.get("dry_run_default", True),
        )

    def check_device_allowed(self, device_id: str) -> bool:
        """Check if device is allowed by policy.

        Args:
            device_id: Device identifier.

        Returns:
            True if device is allowed.
        """
        if not self.policy:
            return False

        if "*" in self.policy.device_allowlist:
            return True

        return device_id in self.policy.device_allowlist

    def check_backend_allowed(self, backend: str) -> bool:
        """Check if backend is allowed by policy.

        Args:
            backend: Backend name.

        Returns:
            True if backend is allowed.
        """
        if not self.policy:
            return False

        return backend not in self.policy.backend_restrictions

    def check_limits(self, device_id: str, setpoint: Dict[str, Any]) -> bool:
        """Check if setpoint is within limits.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary.

        Returns:
            True if setpoint is within limits.
        """
        if not self.policy:
            return False

        limits = self.policy.device_limits.get(device_id)
        if not limits:
            # No limits defined, allow
            return True

        # Check power limits
        power_key = None
        if "power_watts" in setpoint:
            power_key = "power_watts"
        elif "power_limit_watts" in setpoint:
            power_key = "power_limit_watts"

        if power_key:
            power = setpoint[power_key]
            if limits.max_power_watts and power > limits.max_power_watts:
                return False
            if limits.min_power_watts and power < limits.min_power_watts:
                return False

        # Check temperature limits
        if "temp_celsius" in setpoint:
            temp = setpoint["temp_celsius"]
            if limits.max_temp_celsius and temp > limits.max_temp_celsius:
                return False
            if limits.min_temp_celsius and temp < limits.min_temp_celsius:
                return False

        # Check voltage limits
        if "voltage_volts" in setpoint:
            voltage = setpoint["voltage_volts"]
            if limits.max_voltage_volts and voltage > limits.max_voltage_volts:
                return False

        # Check clock limits
        if "clock_mhz" in setpoint:
            clock = setpoint["clock_mhz"]
            if limits.max_clock_mhz and clock > limits.max_clock_mhz:
                return False

        return True

    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded.

        Returns:
            True if rate limit is not exceeded.
        """
        if not self.policy:
            return False

        now = time.time()
        window_start = now - self.policy.rate_limit.window_seconds

        # Remove old entries
        self._command_history = [ts for ts in self._command_history if ts >= window_start]

        # Check rate
        if len(self._command_history) >= self.policy.rate_limit.commands_per_minute:
            return False

        # Record this check
        self._command_history.append(now)
        return True

    def check_approval(self, operation: str) -> bool:
        """Check if operation has required approval.

        Args:
            operation: Operation description.

        Returns:
            True if approval requirements are met.
        """
        if not self.policy:
            return False

        if not self.policy.approval_gate.required:
            return True

        # In a real implementation, this would check an approval database
        # For now, we return False if approval is required
        return False

    def get_environment(self) -> Environment:
        """Get current environment.

        Returns:
            Current environment.
        """
        if not self.policy:
            return Environment.DEV
        return self.policy.environment

    def is_dry_run_default(self) -> bool:
        """Check if dry-run is default.

        Returns:
            True if dry-run is default.
        """
        if not self.policy:
            return True
        return self.policy.dry_run_default
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

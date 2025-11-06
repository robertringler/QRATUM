"""HCAL policy enforcement and validation."""

from typing import Any, Dict

import yaml


class PolicyViolation(Exception):
    """Raised when a policy is violated."""
    pass


class Policy:
    """Hardware control policy."""

    def __init__(self, config: Dict[str, Any]):
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
    def from_file(cls, path: str) -> "Policy":
        """Load policy from YAML file.

        Args:
            path: Path to policy YAML file

        Returns:
            Policy instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def validate_plan(self, plan: Dict[str, Any]) -> None:
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
                    raise PolicyViolation(
                        f"Power limit {power}W exceeds maximum {max_power}W"
                    )

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

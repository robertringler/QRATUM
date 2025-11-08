"""Hardware policy enforcement and validation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class PolicyViolation(Exception):
    """Exception raised when a hardware operation violates policy constraints."""
    pass


@dataclass
class DeviceLimits:
    """Device-specific operational limits."""
    device_id: str
    power_watts_max: Optional[float] = None
    clock_mhz_range: Optional[Tuple[int, int]] = None  # (min, max)
    fan_percent_range: Optional[Tuple[int, int]] = None  # (min, max)
    temp_c_max: Optional[float] = None


class PolicyEngine:
    """
    Hardware policy enforcement engine.

    Validates hardware operations against defined policies and constraints.
    """

    def __init__(self):
        self.device_limits: Dict[str, DeviceLimits] = {}

    def set_device_limits(self, device_id: str, limits: DeviceLimits) -> None:
        """Set operational limits for a device."""
        self.device_limits[device_id] = limits

    def get_device_limits(self, device_id: str) -> DeviceLimits:
        """Get operational limits for a device."""
        # Return default limits if not configured
        if device_id not in self.device_limits:
            return DeviceLimits(device_id=device_id)
        return self.device_limits[device_id]

    def validate_operation(
        self,
        device_id: str,
        operation: str,
        setpoints: Dict[str, Any],
        enable_actuation: bool = True
    ) -> None:
        """
        Validate a hardware operation against policy.

        Args:
            device_id: Device identifier
            operation: Operation type (e.g., 'reconfigure')
            setpoints: Requested device setpoints
            enable_actuation: Whether to enable actual hardware changes

        Raises:
            PolicyViolation: If operation violates policy constraints
        """
        limits = self.get_device_limits(device_id)

        # Validate power limit
        if "power_limit_w" in setpoints and limits.power_watts_max and setpoints["power_limit_w"] > limits.power_watts_max:
            raise PolicyViolation(
                f"Power limit {setpoints['power_limit_w']}W exceeds maximum "
                f"{limits.power_watts_max}W for device {device_id}"
            )

        # Validate clock frequency
        if "sm_clock_mhz" in setpoints and limits.clock_mhz_range:
            min_clk, max_clk = limits.clock_mhz_range
            if not (min_clk <= setpoints["sm_clock_mhz"] <= max_clk):
                raise PolicyViolation(
                    f"Clock frequency {setpoints['sm_clock_mhz']}MHz outside valid range "
                    f"[{min_clk}, {max_clk}]MHz for device {device_id}"
                )

        # Validate fan speed
        if "fan_percent" in setpoints and limits.fan_percent_range:
            min_fan, max_fan = limits.fan_percent_range
            if not (min_fan <= setpoints["fan_percent"] <= max_fan):
                raise PolicyViolation(
                    f"Fan speed {setpoints['fan_percent']}% outside valid range "
                    f"[{min_fan}, {max_fan}]% for device {device_id}"
                )

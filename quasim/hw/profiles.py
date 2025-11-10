"""
Reconfiguration profiles for common hardware optimization scenarios.

Profiles define target setpoints and constraints for specific workload types.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """Hardware reconfiguration profile."""

    name: str
    description: str
    setpoints: Dict[str, Dict[str, Any]]  # device_type -> {param: value}
    constraints: Optional[Dict[str, Any]] = None


# Built-in profiles
PROFILES = {
    "low-latency": Profile(
        name="low-latency",
        description="Maximize clocks, minimize latency for interactive workloads",
        setpoints={
            "gpu": {
                "sm_clock_mhz": 2100,
                "mem_clock_mhz": 2619,
                "power_limit_w": 350,
                "fan_percent": 80,
            },
            "cpu": {
                "governor": "performance",
                "min_freq_mhz": 3000,
            },
        },
        constraints={
            "temp_c_max": 85,
        },
    ),
    "energy-cap": Profile(
        name="energy-cap",
        description="Minimize power consumption for sustained workloads",
        setpoints={
            "gpu": {
                "power_limit_w": 200,
                "sm_clock_mhz": 1410,
                "mem_clock_mhz": 1593,
                "fan_percent": 40,
            },
            "cpu": {
                "governor": "powersave",
                "max_freq_mhz": 2400,
            },
        },
        constraints={
            "power_watts_max": 250,
        },
    ),
    "coherence": Profile(
        name="coherence",
        description="Optimize for quantum coherence and low noise",
        setpoints={
            "gpu": {
                "power_limit_w": 150,
                "sm_clock_mhz": 1200,
                "ecc_enabled": True,
                "fan_percent": 30,
            },
            "cryo": {
                "temp_k": 0.015,  # 15mK
                "bias_mv": 0.0,
            },
        },
        constraints={
            "temp_c_max": 60,
            "power_watts_max": 200,
        },
    ),
    "balanced": Profile(
        name="balanced",
        description="Balanced performance and efficiency",
        setpoints={
            "gpu": {
                "power_limit_w": 250,
                "sm_clock_mhz": 1800,
                "mem_clock_mhz": 2400,
                "fan_percent": 60,
            },
            "cpu": {
                "governor": "schedutil",
            },
        },
    ),
}


class ReconfigurationProfile:
    """
    Hardware reconfiguration profile planner.

    Translates high-level profiles to device-specific setpoints.
    """

    def __init__(self, profile: Profile):
        self.profile = profile

    @classmethod
    def load(cls, profile_name: str) -> "ReconfigurationProfile":
        """Load profile by name."""
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")

        return cls(PROFILES[profile_name])

    @classmethod
    def list_profiles(cls) -> List[str]:
        """List available profile names."""
        return list(PROFILES.keys())

    def plan(
        self,
        topology: Any,  # TopologyDiscovery
        policy: Any,  # PolicyEngine
        devices: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate reconfiguration plan from profile.

        Args:
            topology: Hardware topology
            policy: Policy engine
            devices: Device IDs to configure (None = all)
            constraints: Additional constraints

        Returns:
            Reconfiguration plan ready for actuator
        """
        plan = {
            "profile": self.profile.name,
            "description": self.profile.description,
            "plan_id": f"{self.profile.name}-{id(self)}",
            "devices": {},
            "constraints": self.profile.constraints or {},
        }

        # Merge additional constraints
        if constraints:
            plan["constraints"].update(constraints)

        # Get devices to configure
        target_devices = (
            topology.devices
            if not devices
            else {d: topology.devices[d] for d in devices if d in topology.devices}
        )

        # Map profile setpoints to devices
        for device_id, device_info in target_devices.items():
            device_type = device_info.type

            if device_type in self.profile.setpoints:
                # Get profile setpoints for this device type
                type_setpoints = self.profile.setpoints[device_type].copy()

                # Apply policy limits
                device_limits = policy.get_device_limits(device_id)
                type_setpoints = self._apply_limits(type_setpoints, device_limits)

                plan["devices"][device_id] = type_setpoints

        # Validate plan against policy
        self._validate_plan(plan, policy)

        logger.info(f"Generated plan '{self.profile.name}' for {len(plan['devices'])} devices")
        return plan

    def _apply_limits(
        self, setpoints: Dict[str, Any], limits: Any  # DeviceLimits
    ) -> Dict[str, Any]:
        """Apply policy limits to setpoints."""
        limited = setpoints.copy()

        # Power limit
        if "power_limit_w" in limited and limits.power_watts_max:
            limited["power_limit_w"] = min(limited["power_limit_w"], limits.power_watts_max)

        # Clock limits
        if "sm_clock_mhz" in limited and limits.clock_mhz_range:
            min_clk, max_clk = limits.clock_mhz_range
            limited["sm_clock_mhz"] = max(min_clk, min(limited["sm_clock_mhz"], max_clk))

        # Fan limits
        if "fan_percent" in limited and limits.fan_percent_range:
            min_fan, max_fan = limits.fan_percent_range
            limited["fan_percent"] = max(min_fan, min(limited["fan_percent"], max_fan))

        return limited

    def _validate_plan(self, plan: Dict[str, Any], policy: Any) -> None:
        """Validate plan against policy (raises PolicyViolation if invalid)."""
        for device_id, setpoints in plan["devices"].items():
            policy.validate_operation(
                device_id=device_id,
                operation="reconfigure",
                setpoints=setpoints,
                enable_actuation=False,  # Validation only
            )


def create_custom_profile(
    name: str,
    description: str,
    setpoints: Dict[str, Dict[str, Any]],
    constraints: Optional[Dict[str, Any]] = None,
) -> Profile:
    """
    Create custom reconfiguration profile.

    Example:
        profile = create_custom_profile(
            name="my-profile",
            description="Custom optimization",
            setpoints={
                "gpu": {"power_limit_w": 275, "sm_clock_mhz": 1950},
            }
        )
    """
    return Profile(
        name=name,
        description=description,
        setpoints=setpoints,
        constraints=constraints,
    )

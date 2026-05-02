"""Reconfiguration profiles for HCAL."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProfileType(Enum):
    """Reconfiguration profile types."""

    LOW_LATENCY = "low-latency"
    ENERGY_CAP = "energy-cap"
    COHERENCE = "coherence"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class ReconfigProfile:
    """Reconfiguration profile."""

    name: str
    profile_type: ProfileType
    description: str
    setpoints: Dict[str, Dict[str, Any]]
    constraints: Dict[str, Any]


class ProfileManager:
    """Manage reconfiguration profiles."""

    def __init__(self):
        """Initialize profile manager."""

        self.profiles = {}
        self._load_default_profiles()

    def _load_default_profiles(self):
        """Load default optimization profiles."""

        # Low-latency profile: Maximize performance
        self.profiles["low-latency"] = ReconfigProfile(
            name="low-latency",
            profile_type=ProfileType.LOW_LATENCY,
            description="Maximize performance for low-latency workloads",
            setpoints={
                "gpu": {
                    "power_limit_watts": 300.0,
                    "sm_clock_mhz": 1800,
                    "mem_clock_mhz": 6000,
                }
            },
            constraints={"max_temp_celsius": 85.0},
        )

        # Energy-cap profile: Minimize power
        self.profiles["energy-cap"] = ReconfigProfile(
            name="energy-cap",
            profile_type=ProfileType.ENERGY_CAP,
            description="Minimize power consumption",
            setpoints={
                "gpu": {
                    "power_limit_watts": 150.0,
                    "sm_clock_mhz": 1200,
                    "mem_clock_mhz": 4000,
                }
            },
            constraints={"max_power_watts": 150.0},
        )

        # Coherence profile: Optimize for quantum coherence
        self.profiles["coherence"] = ReconfigProfile(
            name="coherence",
            profile_type=ProfileType.COHERENCE,
            description="Optimize for quantum coherence times",
            setpoints={
                "gpu": {
                    "power_limit_watts": 200.0,
                    "sm_clock_mhz": 1500,
                    "mem_clock_mhz": 5000,
                }
            },
            constraints={"max_temp_celsius": 70.0},
        )

        # Balanced profile: Balance performance and efficiency
        self.profiles["balanced"] = ReconfigProfile(
            name="balanced",
            profile_type=ProfileType.BALANCED,
            description="Balance performance and efficiency",
            setpoints={
                "gpu": {
                    "power_limit_watts": 225.0,
                    "sm_clock_mhz": 1600,
                    "mem_clock_mhz": 5500,
                }
            },
            constraints={
                "max_temp_celsius": 80.0,
                "max_power_watts": 250.0,
            },
        )

    def get_profile(self, name: str) -> Optional[ReconfigProfile]:
        """Get profile by name.

        Args:
            name: Profile name.

        Returns:
            ReconfigProfile instance or None.
        """

        return self.profiles.get(name)

    def create_custom_profile(
        self,
        name: str,
        description: str,
        setpoints: Dict[str, Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReconfigProfile:
        """Create a custom profile.

        Args:
            name: Profile name.
            description: Profile description.
            setpoints: Setpoints dictionary.
            constraints: Constraints dictionary.

        Returns:
            ReconfigProfile instance.
        """

        profile = ReconfigProfile(
            name=name,
            profile_type=ProfileType.CUSTOM,
            description=description,
            setpoints=setpoints,
            constraints=constraints or {},
        )

        self.profiles[name] = profile
        return profile

    def list_profiles(self) -> List[str]:
        """List all available profiles.

        Returns:
            List of profile names.
        """

        return list(self.profiles.keys())

    def apply_profile(
        self, profile_name: str, device_type: str, device_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get setpoints for applying a profile to a device.

        Args:
            profile_name: Profile name.
            device_type: Device type (e.g., 'gpu', 'cpu').
            device_id: Device identifier.

        Returns:
            Setpoints dictionary or None.
        """

        profile = self.get_profile(profile_name)
        if not profile:
            return None

        if device_type not in profile.setpoints:
            return None

        return profile.setpoints[device_type].copy()

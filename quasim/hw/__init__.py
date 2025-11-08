"""Hardware management and reconfiguration for QuASIM.

This module provides hardware topology discovery, policy enforcement,
and reconfiguration profiles for optimizing hardware settings based on
workload characteristics.
"""

from quasim.hw.policy import (
    DeviceLimits,
    PolicyEngine,
    PolicyViolation,
)
from quasim.hw.profiles import (
    PROFILES,
    Profile,
    ReconfigurationProfile,
    create_custom_profile,
)
from quasim.hw.topology import (
    DeviceInfo,
    TopologyDiscovery,
)

__all__ = [
    "Profile",
    "ReconfigurationProfile",
    "PROFILES",
    "create_custom_profile",
    "DeviceInfo",
    "TopologyDiscovery",
    "DeviceLimits",
    "PolicyEngine",
    "PolicyViolation",
]

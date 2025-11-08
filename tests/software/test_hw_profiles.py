"""Tests for hardware reconfiguration profiles."""

from __future__ import annotations

import pytest

from quasim.hw import (
    PROFILES,
    DeviceLimits,
    PolicyEngine,
    Profile,
    ReconfigurationProfile,
    TopologyDiscovery,
    create_custom_profile,
)


def test_profile_dataclass():
    """Test Profile dataclass creation."""
    profile = Profile(
        name="test-profile",
        description="Test profile",
        setpoints={"gpu": {"power_limit_w": 250}},
        constraints={"temp_c_max": 80}
    )

    assert profile.name == "test-profile"
    assert profile.description == "Test profile"
    assert profile.setpoints == {"gpu": {"power_limit_w": 250}}
    assert profile.constraints == {"temp_c_max": 80}


def test_builtin_profiles_exist():
    """Test that all expected built-in profiles exist."""
    expected_profiles = ["low-latency", "energy-cap", "coherence", "balanced"]

    for profile_name in expected_profiles:
        assert profile_name in PROFILES
        profile = PROFILES[profile_name]
        assert isinstance(profile, Profile)
        assert profile.name == profile_name
        assert len(profile.description) > 0
        assert len(profile.setpoints) > 0


def test_low_latency_profile():
    """Test low-latency profile configuration."""
    profile = PROFILES["low-latency"]

    assert profile.name == "low-latency"
    assert "gpu" in profile.setpoints
    assert "cpu" in profile.setpoints
    assert profile.setpoints["gpu"]["power_limit_w"] == 350
    assert profile.setpoints["gpu"]["sm_clock_mhz"] == 2100
    assert profile.constraints["temp_c_max"] == 85


def test_energy_cap_profile():
    """Test energy-cap profile configuration."""
    profile = PROFILES["energy-cap"]

    assert profile.name == "energy-cap"
    assert "gpu" in profile.setpoints
    assert "cpu" in profile.setpoints
    assert profile.setpoints["gpu"]["power_limit_w"] == 200
    assert profile.setpoints["cpu"]["governor"] == "powersave"
    assert profile.constraints["power_watts_max"] == 250


def test_coherence_profile():
    """Test coherence profile configuration."""
    profile = PROFILES["coherence"]

    assert profile.name == "coherence"
    assert "gpu" in profile.setpoints
    assert "cryo" in profile.setpoints
    assert profile.setpoints["gpu"]["ecc_enabled"] is True
    assert profile.setpoints["cryo"]["temp_k"] == 0.015
    assert profile.constraints["temp_c_max"] == 60


def test_balanced_profile():
    """Test balanced profile configuration."""
    profile = PROFILES["balanced"]

    assert profile.name == "balanced"
    assert "gpu" in profile.setpoints
    assert "cpu" in profile.setpoints
    assert profile.setpoints["gpu"]["power_limit_w"] == 250
    assert profile.setpoints["cpu"]["governor"] == "schedutil"


def test_reconfiguration_profile_load():
    """Test loading a profile by name."""
    reconfig = ReconfigurationProfile.load("low-latency")

    assert isinstance(reconfig, ReconfigurationProfile)
    assert reconfig.profile.name == "low-latency"


def test_reconfiguration_profile_load_invalid():
    """Test loading an invalid profile name."""
    with pytest.raises(ValueError) as exc_info:
        ReconfigurationProfile.load("nonexistent-profile")

    assert "Unknown profile" in str(exc_info.value)
    assert "nonexistent-profile" in str(exc_info.value)


def test_list_profiles():
    """Test listing available profile names."""
    profiles = ReconfigurationProfile.list_profiles()

    assert isinstance(profiles, list)
    assert len(profiles) >= 4
    assert "low-latency" in profiles
    assert "energy-cap" in profiles
    assert "coherence" in profiles
    assert "balanced" in profiles


def test_plan_generation_basic():
    """Test basic plan generation."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("gpu0", "gpu", "NVIDIA A100")
    topology.add_device("cpu0", "cpu", "Intel Xeon")

    # Setup policy
    policy = PolicyEngine()
    policy.set_device_limits("gpu0", DeviceLimits(
        device_id="gpu0",
        power_watts_max=400,
        clock_mhz_range=(1000, 2500),
        fan_percent_range=(0, 100)
    ))
    policy.set_device_limits("cpu0", DeviceLimits(device_id="cpu0"))

    # Generate plan
    reconfig = ReconfigurationProfile.load("low-latency")
    plan = reconfig.plan(topology, policy)

    assert plan["profile"] == "low-latency"
    assert "plan_id" in plan
    assert "devices" in plan
    assert "constraints" in plan
    assert "gpu0" in plan["devices"]
    assert "cpu0" in plan["devices"]
    assert plan["devices"]["gpu0"]["power_limit_w"] == 350
    assert plan["devices"]["gpu0"]["sm_clock_mhz"] == 2100


def test_plan_generation_with_device_filter():
    """Test plan generation with device filtering."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("gpu0", "gpu", "NVIDIA A100")
    topology.add_device("gpu1", "gpu", "NVIDIA A100")
    topology.add_device("cpu0", "cpu", "Intel Xeon")

    # Setup policy
    policy = PolicyEngine()
    for device_id in ["gpu0", "gpu1", "cpu0"]:
        policy.set_device_limits(device_id, DeviceLimits(
            device_id=device_id,
            power_watts_max=400,
            clock_mhz_range=(1000, 2500),
            fan_percent_range=(0, 100)
        ))

    # Generate plan for only gpu0
    reconfig = ReconfigurationProfile.load("low-latency")
    plan = reconfig.plan(topology, policy, devices=["gpu0"])

    assert "gpu0" in plan["devices"]
    assert "gpu1" not in plan["devices"]
    assert "cpu0" not in plan["devices"]


def test_plan_generation_with_additional_constraints():
    """Test plan generation with additional constraints."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("gpu0", "gpu", "NVIDIA A100")

    # Setup policy
    policy = PolicyEngine()
    policy.set_device_limits("gpu0", DeviceLimits(
        device_id="gpu0",
        power_watts_max=400,
        clock_mhz_range=(1000, 2500),
        fan_percent_range=(0, 100)
    ))

    # Generate plan with additional constraints
    reconfig = ReconfigurationProfile.load("low-latency")
    additional_constraints = {"power_watts_max": 300}
    plan = reconfig.plan(topology, policy, constraints=additional_constraints)

    assert plan["constraints"]["temp_c_max"] == 85  # From profile
    assert plan["constraints"]["power_watts_max"] == 300  # Additional


def test_apply_limits_power():
    """Test applying power limits to setpoints via public API."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("test", "gpu", "NVIDIA A100")

    # Setup policy with power limit
    policy = PolicyEngine()
    policy.set_device_limits("test", DeviceLimits(
        device_id="test",
        power_watts_max=350
    ))

    # Create a custom profile with high setpoints
    profile = Profile(
        name="test-profile",
        description="Test profile",
        setpoints={"gpu": {"power_limit_w": 400, "sm_clock_mhz": 2000}},
        constraints={}
    )
    reconfig = create_custom_profile(profile)

    plan = reconfig.plan(topology, policy)
    setpoints = plan["devices"]["test"]["setpoints"]

    assert setpoints["power_limit_w"] == 350  # Capped to max
    assert setpoints["sm_clock_mhz"] == 2000  # Unchanged


def test_apply_limits_clock():
    """Test applying clock limits to setpoints via public API."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("test", "gpu", "NVIDIA A100")

    # Setup policy with clock range
    policy = PolicyEngine()
    policy.set_device_limits("test", DeviceLimits(
        device_id="test",
        clock_mhz_range=(1000, 2200)
    ))

    # Create a custom profile with high clock setpoint
    profile = Profile(
        name="test-profile",
        description="Test profile",
        setpoints={"gpu": {"sm_clock_mhz": 2500}},
        constraints={}
    )
    reconfig = create_custom_profile(profile)

    plan = reconfig.plan(topology, policy)
    setpoints = plan["devices"]["test"]["setpoints"]

    assert setpoints["sm_clock_mhz"] == 2200  # Capped to max


def test_apply_limits_clock_min():
    """Test applying minimum clock limit to setpoints."""
    reconfig = ReconfigurationProfile.load("low-latency")

    setpoints = {"sm_clock_mhz": 800}
    limits = DeviceLimits(
        device_id="test",
        clock_mhz_range=(1000, 2200)
    )

    limited = reconfig._apply_limits(setpoints, limits)

    assert limited["sm_clock_mhz"] == 1000  # Raised to min


def test_apply_limits_fan():
    """Test applying fan speed limits to setpoints."""
    reconfig = ReconfigurationProfile.load("low-latency")

    setpoints = {"fan_percent": 90}
    limits = DeviceLimits(
        device_id="test",
        fan_percent_range=(30, 80)
    )

    limited = reconfig._apply_limits(setpoints, limits)

    assert limited["fan_percent"] == 80  # Capped to max


def test_validate_plan_valid():
    """Test validation of a valid plan."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("gpu0", "gpu", "NVIDIA A100")

    # Setup policy with generous limits
    policy = PolicyEngine()
    policy.set_device_limits("gpu0", DeviceLimits(
        device_id="gpu0",
        power_watts_max=400,
        clock_mhz_range=(1000, 2500),
        fan_percent_range=(0, 100)
    ))

    # Generate and validate plan
    reconfig = ReconfigurationProfile.load("low-latency")
    plan = reconfig.plan(topology, policy)

    # Should not raise exception
    assert plan is not None


def test_validate_plan_policy_violation():
    """Test validation with policy violation."""
    # Setup topology
    topology = TopologyDiscovery()
    topology.add_device("gpu0", "gpu", "NVIDIA A100")

    # Setup policy with restrictive limits
    policy = PolicyEngine()
    policy.set_device_limits("gpu0", DeviceLimits(
        device_id="gpu0",
        power_watts_max=300,  # Less than low-latency profile wants (350)
        clock_mhz_range=(1000, 2000),  # Less than low-latency profile wants (2100)
        fan_percent_range=(0, 100)
    ))

    # Generate plan - should apply limits and pass validation
    reconfig = ReconfigurationProfile.load("low-latency")
    plan = reconfig.plan(topology, policy)

    # Verify limits were applied
    assert plan["devices"]["gpu0"]["power_limit_w"] == 300
    assert plan["devices"]["gpu0"]["sm_clock_mhz"] == 2000


def test_create_custom_profile():
    """Test creating a custom profile."""
    profile = create_custom_profile(
        name="my-profile",
        description="Custom optimization",
        setpoints={
            "gpu": {"power_limit_w": 275, "sm_clock_mhz": 1950},
        },
        constraints={"temp_c_max": 75}
    )

    assert isinstance(profile, Profile)
    assert profile.name == "my-profile"
    assert profile.description == "Custom optimization"
    assert profile.setpoints["gpu"]["power_limit_w"] == 275
    assert profile.constraints["temp_c_max"] == 75


def test_custom_profile_without_constraints():
    """Test creating a custom profile without constraints."""
    profile = create_custom_profile(
        name="simple-profile",
        description="Simple custom profile",
        setpoints={
            "gpu": {"power_limit_w": 250},
        }
    )

    assert isinstance(profile, Profile)
    assert profile.name == "simple-profile"
    assert profile.constraints is None


def test_plan_with_missing_device_type():
    """Test plan generation when device type not in profile."""
    # Setup topology with device type not in profile
    topology = TopologyDiscovery()
    topology.add_device("fpga0", "fpga", "Xilinx FPGA")

    # Setup policy
    policy = PolicyEngine()
    policy.set_device_limits("fpga0", DeviceLimits(device_id="fpga0"))

    # Generate plan
    reconfig = ReconfigurationProfile.load("low-latency")
    plan = reconfig.plan(topology, policy)

    # FPGA should not be in plan since it's not in the profile
    assert "fpga0" not in plan["devices"]
    assert len(plan["devices"]) == 0

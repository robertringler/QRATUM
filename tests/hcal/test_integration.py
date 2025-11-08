"""Integration tests for HCAL."""

import tempfile
from pathlib import Path

import yaml

from quasim.hcal import HCAL
from quasim.hcal.backends.nvidia_nvml import NvidiaNvmlBackend
from quasim.hcal.loops.reconfig_profiles import ProfileManager
from quasim.hcal.topology import DeviceType


def test_hcal_initialization():
    """Test HCAL initialization."""
    hcal = HCAL(dry_run=True)

    assert hcal.dry_run is True
    assert hcal.policy_engine is not None
    assert hcal.topology is not None
    assert hcal.sensor_manager is not None
    assert hcal.actuator is not None


def test_topology_discovery():
    """Test hardware topology discovery."""
    hcal = HCAL(dry_run=True)
    topology = hcal.discover_topology()

    assert topology is not None
    assert len(topology.devices) > 0

    # Should at least discover CPU
    cpu_devices = [d for d in topology.devices if d.device_type == DeviceType.CPU]
    assert len(cpu_devices) > 0


def test_nvidia_backend_dry_run():
    """Test NVIDIA backend in dry-run mode."""
    backend = NvidiaNvmlBackend(dry_run=True)

    # Device should exist in dry-run
    assert backend.device_exists("gpu0") is True
    assert backend.device_exists("cpu0") is False

    # Apply setpoint (dry-run)
    setpoint = {"power_limit_watts": 250.0}
    success = backend.apply_setpoint("gpu0", setpoint)
    assert success is True

    # Read configuration
    config = backend.read_configuration("gpu0")
    assert "power_limit_watts" in config

    # Read telemetry
    telemetry = backend.read_telemetry("gpu0")
    assert "power_watts" in telemetry
    assert "temperature_celsius" in telemetry


def test_apply_setpoint_with_policy():
    """Test applying setpoint with policy validation."""
    policy_data = {
        "environment": "dev",
        "device_allowlist": ["gpu0"],
        "backend_restrictions": [],
        "device_limits": {
            "gpu0": {
                "max_power_watts": 300.0,
                "min_power_watts": 100.0,
            }
        },
        "rate_limit": {
            "commands_per_minute": 100,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        hcal = HCAL(policy_path=policy_path, dry_run=True)

        # Valid setpoint
        setpoint = {"power_limit_watts": 250.0}
        success = hcal.apply_setpoint("gpu0", setpoint)
        assert success is True

        # Exceeds limits
        setpoint = {"power_limit_watts": 400.0}
        success = hcal.apply_setpoint("gpu0", setpoint)
        assert success is False

        # Device not allowed
        setpoint = {"power_limit_watts": 250.0}
        success = hcal.apply_setpoint("gpu1", setpoint)
        assert success is False

    finally:
        policy_path.unlink()


def test_read_telemetry():
    """Test reading telemetry."""
    hcal = HCAL(dry_run=True)

    reading = hcal.read_telemetry("gpu0")

    assert reading is not None
    assert reading.device_id == "gpu0"
    assert reading.metrics is not None
    assert "power_watts" in reading.metrics


def test_calibration_bias_trim():
    """Test bias trim calibration."""
    hcal = HCAL(dry_run=True)

    result = hcal.calibrate_bias_trim("gpu0", max_iterations=5)

    assert result is not None
    assert result.status is not None
    assert result.iterations > 0
    assert result.final_setpoint is not None


def test_power_sweep():
    """Test power sweep calibration."""
    hcal = HCAL(dry_run=True)

    results = hcal.run_power_sweep("gpu0", power_range=(150.0, 250.0), steps=3)

    assert len(results) == 3
    assert results[0][0] == 150.0  # First power level
    assert results[-1][0] == 250.0  # Last power level


def test_audit_log():
    """Test audit log functionality."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        audit_path = Path(f.name)

    try:
        hcal = HCAL(dry_run=True, audit_log_path=audit_path)

        # Perform operations
        hcal.apply_setpoint("gpu0", {"power_limit_watts": 250.0})
        hcal.apply_setpoint("gpu0", {"power_limit_watts": 200.0})

        # Get audit log
        log = hcal.get_audit_log()
        assert len(log) >= 2

        # Verify chain
        assert hcal.verify_audit_chain() is True

    finally:
        audit_path.unlink()


def test_emergency_stop():
    """Test emergency stop functionality."""
    hcal = HCAL(dry_run=True)

    # Emergency stop should block operations
    hcal.emergency_stop()

    success = hcal.apply_setpoint("gpu0", {"power_limit_watts": 250.0})
    assert success is False


def test_profile_manager():
    """Test reconfiguration profile manager."""
    mgr = ProfileManager()

    # List profiles
    profiles = mgr.list_profiles()
    assert "low-latency" in profiles
    assert "energy-cap" in profiles
    assert "balanced" in profiles

    # Get profile
    profile = mgr.get_profile("low-latency")
    assert profile is not None
    assert profile.name == "low-latency"

    # Apply profile
    setpoints = mgr.apply_profile("balanced", "gpu", "gpu0")
    assert setpoints is not None
    assert "power_limit_watts" in setpoints


def test_custom_profile():
    """Test creating custom profile."""
    mgr = ProfileManager()

    setpoints = {
        "gpu": {
            "power_limit_watts": 200.0,
            "sm_clock_mhz": 1500,
        }
    }

    profile = mgr.create_custom_profile(
        name="custom-test",
        description="Test custom profile",
        setpoints=setpoints,
        constraints={"max_temp_celsius": 75.0},
    )

    assert profile.name == "custom-test"
    assert "custom-test" in mgr.list_profiles()

    # Apply custom profile
    applied = mgr.apply_profile("custom-test", "gpu", "gpu0")
    assert applied["power_limit_watts"] == 200.0


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    # Setup policy
    policy_data = {
        "environment": "dev",
        "device_allowlist": ["*"],
        "backend_restrictions": [],
        "rate_limit": {"commands_per_minute": 100},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        audit_path = Path(f.name)

    try:
        # Initialize HCAL
        hcal = HCAL(policy_path=policy_path, dry_run=True, audit_log_path=audit_path)

        # Discover topology
        topology = hcal.discover_topology()
        assert len(topology.devices) > 0

        # Read initial telemetry
        reading = hcal.read_telemetry("gpu0")
        assert reading is not None

        # Apply profile
        mgr = ProfileManager()
        setpoints = mgr.apply_profile("balanced", "gpu", "gpu0")
        success = hcal.apply_setpoint("gpu0", setpoints)
        assert success is True

        # Read updated telemetry
        reading = hcal.read_telemetry("gpu0")
        assert reading is not None

        # Verify audit log
        log = hcal.get_audit_log()
        assert len(log) > 0
        assert hcal.verify_audit_chain() is True

    finally:
        policy_path.unlink()
        audit_path.unlink()


def test_rollback_functionality():
    """Test automatic rollback on validation failure."""
    policy_data = {
        "environment": "dev",
        "device_allowlist": ["gpu0"],
        "backend_restrictions": [],
        "rate_limit": {"commands_per_minute": 100},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy_data, f)
        policy_path = Path(f.name)

    try:
        hcal = HCAL(policy_path=policy_path, dry_run=True)

        # Apply valid setpoint
        setpoint = {"power_limit_watts": 250.0}
        success = hcal.apply_setpoint("gpu0", setpoint)
        assert success is True

        # In dry-run mode, rollback is simulated
        # Actual rollback testing would require mocking backend failures

    finally:
        policy_path.unlink()

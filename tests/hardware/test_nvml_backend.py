"""Tests for NVIDIA NVML backend."""

import sys
from unittest.mock import MagicMock

import pytest

# Create a mock pynvml module before importing the backend
mock_pynvml_module = MagicMock()
mock_pynvml_module.NVML_TEMPERATURE_GPU = 0
mock_pynvml_module.NVML_CLOCK_SM = 0
mock_pynvml_module.NVML_CLOCK_MEM = 1
mock_pynvml_module.NVML_FEATURE_ENABLED = 1
mock_pynvml_module.NVML_FEATURE_DISABLED = 0

sys.modules["pynvml"] = mock_pynvml_module

# Import after mocking
from quasim.hardware import nvml_backend


@pytest.fixture(autouse=True)
def reset_mock():
    """Reset mock calls and side effects before each test."""
    mock_pynvml_module.reset_mock()
    # Clear all side effects for better test isolation
    mock_pynvml_module.nvmlInit.side_effect = None
    mock_pynvml_module.nvmlDeviceGetHandleByIndex.side_effect = None
    mock_pynvml_module.nvmlDeviceGetPowerUsage.side_effect = None
    mock_pynvml_module.nvmlDeviceGetPowerManagementLimit.side_effect = None
    mock_pynvml_module.nvmlDeviceGetTemperature.side_effect = None
    mock_pynvml_module.nvmlDeviceGetClockInfo.side_effect = None
    mock_pynvml_module.nvmlDeviceGetFanSpeed.side_effect = None
    mock_pynvml_module.nvmlDeviceGetEccMode.side_effect = None
    mock_pynvml_module.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = None
    yield


@pytest.fixture
def mock_pynvml():
    """Get the mock pynvml module."""
    yield mock_pynvml_module


class TestNVMLBackendInitialization:
    """Test NVML backend initialization."""

    def test_init_success(self, mock_pynvml):
        """Test successful initialization."""
        backend = nvml_backend.NVMLBackend()
        assert backend.initialized is True
        mock_pynvml.nvmlInit.assert_called_once()

    def test_init_failure(self, mock_pynvml):
        """Test initialization failure."""
        mock_pynvml.nvmlInit.side_effect = Exception("NVML init failed")
        backend = nvml_backend.NVMLBackend()
        assert backend.initialized is False

    def test_init_without_pynvml(self):
        """Test initialization when pynvml is not available."""
        original_available = nvml_backend.NVML_AVAILABLE
        nvml_backend.NVML_AVAILABLE = False
        backend = nvml_backend.NVMLBackend()
        assert backend.initialized is False
        nvml_backend.NVML_AVAILABLE = original_available

    def test_shutdown_on_delete(self, mock_pynvml):
        """Test NVML shutdown on object deletion."""
        backend = nvml_backend.NVMLBackend()
        backend.__del__()
        mock_pynvml.nvmlShutdown.assert_called()


class TestListDevices:
    """Test device listing functionality."""

    def test_list_devices_success(self, mock_pynvml):
        """Test successful device listing."""
        # Setup mocks
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handle1 = MagicMock()
        mock_handle2 = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [mock_handle1, mock_handle2]
        mock_pynvml.nvmlDeviceGetName.side_effect = ["Tesla V100", "Tesla T4"]
        mock_pynvml.nvmlDeviceGetSerial.side_effect = ["SN001", "SN002"]

        backend = nvml_backend.NVMLBackend()
        devices = backend.list_devices()

        assert len(devices) == 2
        assert devices[0]["id"] == "GPU0"
        assert devices[0]["name"] == "Tesla V100"
        assert devices[0]["serial"] == "SN001"
        assert devices[1]["id"] == "GPU1"
        assert devices[1]["name"] == "Tesla T4"
        assert devices[1]["serial"] == "SN002"

    def test_list_devices_not_initialized(self, mock_pynvml):
        """Test listing devices when not initialized."""
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")
        backend = nvml_backend.NVMLBackend()
        devices = backend.list_devices()
        assert devices == []

    def test_list_devices_error(self, mock_pynvml):
        """Test error handling in device listing."""
        mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("Device count failed")
        backend = nvml_backend.NVMLBackend()
        devices = backend.list_devices()
        assert devices == []


class TestGetState:
    """Test device state retrieval."""

    def test_get_state_success(self, mock_pynvml):
        """Test successful state retrieval."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W in mW
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 300000  # 300W
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [1530, 877]  # SM, MEM
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 50
        mock_pynvml.nvmlDeviceGetEccMode.return_value = (1, 1)  # enabled

        backend = nvml_backend.NVMLBackend()
        state = backend.get_state("GPU0")

        assert state["power_mw"] == 250000
        assert state["power_limit_mw"] == 300000
        assert state["temp_c"] == 65
        assert state["sm_clock_mhz"] == 1530
        assert state["mem_clock_mhz"] == 877
        assert state["fan_percent"] == 50
        assert state["ecc_enabled"] is True

    def test_get_state_not_initialized(self, mock_pynvml):
        """Test state retrieval when not initialized."""
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")
        backend = nvml_backend.NVMLBackend()
        state = backend.get_state("GPU0")
        assert state == {}

    def test_get_state_partial_failure(self, mock_pynvml):
        """Test state retrieval with partial failures."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Power failed")
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.side_effect = Exception("Power limit failed")
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 70
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = Exception("Clock failed")
        mock_pynvml.nvmlDeviceGetFanSpeed.side_effect = Exception("Fan failed")
        mock_pynvml.nvmlDeviceGetEccMode.side_effect = Exception("ECC failed")

        backend = nvml_backend.NVMLBackend()
        state = backend.get_state("GPU1")

        assert state["power_mw"] is None
        assert state["power_limit_mw"] is None
        assert state["temp_c"] == 70
        assert state["sm_clock_mhz"] is None
        assert state["mem_clock_mhz"] is None
        assert state["fan_percent"] is None
        assert state["ecc_enabled"] is None

    def test_get_state_complete_failure(self, mock_pynvml):
        """Test state retrieval with complete failure."""
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Handle failed")
        backend = nvml_backend.NVMLBackend()
        state = backend.get_state("GPU0")
        assert state == {}


class TestApplySetpoint:
    """Test setpoint application."""

    def test_apply_power_limit_dry_run(self, mock_pynvml):
        """Test power limit application in dry run mode."""
        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["device"] == "GPU0"
        assert result["parameter"] == "power_limit_w"
        assert result["value"] == 250
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()

    def test_apply_power_limit_actual(self, mock_pynvml):
        """Test actual power limit application."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=False)

        assert result["success"] is True
        assert result["dry_run"] is False
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_called_with(mock_handle, 250000)

    def test_apply_sm_clock(self, mock_pynvml):
        """Test SM clock application."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "sm_clock_mhz", 1500, dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceSetApplicationsClocks.assert_called_with(
            mock_handle, mem_clock=0, sm_clock=1500
        )

    def test_apply_mem_clock(self, mock_pynvml):
        """Test memory clock application."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "mem_clock_mhz", 877, dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceSetApplicationsClocks.assert_called_with(
            mock_handle, mem_clock=877, sm_clock=0
        )

    def test_apply_fan_speed(self, mock_pynvml):
        """Test fan speed application."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "fan_percent", 75, dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceSetFanSpeed_v2.assert_called_with(mock_handle, 0, 75)

    def test_apply_ecc_enabled(self, mock_pynvml):
        """Test ECC enable."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "ecc_enabled", True, dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceSetEccMode.assert_called_with(mock_handle, 1)

    def test_apply_ecc_disabled(self, mock_pynvml):
        """Test ECC disable."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "ecc_enabled", False, dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceSetEccMode.assert_called_with(mock_handle, 0)

    def test_apply_unknown_parameter(self, mock_pynvml):
        """Test unknown parameter."""
        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "unknown_param", 100, dry_run=False)

        assert result["success"] is False
        assert "Unknown parameter" in result["error"]

    def test_apply_not_initialized(self, mock_pynvml):
        """Test setpoint application when not initialized."""
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")
        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "power_limit_w", 250)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    def test_apply_setpoint_error(self, mock_pynvml):
        """Test error handling in setpoint application."""
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Apply failed")
        backend = nvml_backend.NVMLBackend()
        result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=False)

        assert result["success"] is False
        assert "Apply failed" in result["error"]


class TestResetToDefaults:
    """Test reset to defaults functionality."""

    def test_reset_dry_run(self, mock_pynvml):
        """Test reset in dry run mode."""
        backend = nvml_backend.NVMLBackend()
        result = backend.reset_to_defaults("GPU0", dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["device"] == "GPU0"
        mock_pynvml.nvmlDeviceResetApplicationsClocks.assert_not_called()

    def test_reset_actual(self, mock_pynvml):
        """Test actual reset."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000

        backend = nvml_backend.NVMLBackend()
        result = backend.reset_to_defaults("GPU1", dry_run=False)

        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["device"] == "GPU1"
        mock_pynvml.nvmlDeviceResetApplicationsClocks.assert_called_with(mock_handle)
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_called_with(mock_handle, 300000)

    def test_reset_power_limit_failure(self, mock_pynvml):
        """Test reset with power limit failure (should continue)."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = Exception(
            "Power limit failed"
        )

        backend = nvml_backend.NVMLBackend()
        result = backend.reset_to_defaults("GPU0", dry_run=False)

        assert result["success"] is True
        mock_pynvml.nvmlDeviceResetApplicationsClocks.assert_called()

    def test_reset_not_initialized(self, mock_pynvml):
        """Test reset when not initialized."""
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")
        backend = nvml_backend.NVMLBackend()
        result = backend.reset_to_defaults("GPU0")
        assert result["success"] is False
        assert "not initialized" in result["error"]

    def test_reset_error(self, mock_pynvml):
        """Test error handling in reset."""
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Reset failed")
        backend = nvml_backend.NVMLBackend()
        result = backend.reset_to_defaults("GPU0", dry_run=False)

        assert result["success"] is False
        assert "Reset failed" in result["error"]


class TestNVMLAvailableFlag:
    """Test NVML_AVAILABLE flag."""

    def test_nvml_available_import(self):
        """Test that NVML_AVAILABLE is exported."""
        from quasim.hardware import NVML_AVAILABLE

        # Should be True since we mocked it
        assert isinstance(NVML_AVAILABLE, bool)
        assert NVML_AVAILABLE is True

    def test_backend_import(self):
        """Test that NVMLBackend can be imported."""
        from quasim.hardware import NVMLBackend

        assert NVMLBackend is not None

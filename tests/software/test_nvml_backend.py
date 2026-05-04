"""Tests for NVIDIA NVML backend."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock pynvml before importing the backend
mock_pynvml = MagicMock()
sys.modules["pynvml"] = mock_pynvml


class TestNVMLBackendWithoutNVML(unittest.TestCase):
    """Test NVMLBackend behavior when pynvml is not available."""

    def _get_nvml_backend(self):
        """Helper to patch NVML_AVAILABLE and return a new NVMLBackend instance."""

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", False):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            return NVMLBackend()

    def test_initialization_without_pynvml(self):
        """Test backend initialization when pynvml is not available."""

        backend = self._get_nvml_backend()
        assert not backend.initialized

    def test_list_devices_without_pynvml(self):
        """Test list_devices returns empty list when not initialized."""

        backend = self._get_nvml_backend()
        devices = backend.list_devices()
        assert devices == []

    def test_get_state_without_pynvml(self):
        """Test get_state returns empty dict when not initialized."""

        backend = self._get_nvml_backend()
        state = backend.get_state("GPU0")
        assert state == {}

    def test_apply_setpoint_without_pynvml(self):
        """Test apply_setpoint returns error when not initialized."""

        backend = self._get_nvml_backend()
        result = backend.apply_setpoint("GPU0", "power_limit_w", 250)
        assert not result["success"]
        assert "error" in result


class TestNVMLBackendWithMock(unittest.TestCase):
    """Test NVMLBackend with mocked pynvml."""

    def setUp(self):
        """Set up mocked pynvml for testing."""

        # Reset mock between tests
        mock_pynvml.reset_mock()
        # Clear all side effects
        mock_pynvml.nvmlInit.side_effect = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = None
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = None
        mock_pynvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = None

        self.mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = self.mock_handle

    def test_initialization_success(self):
        """Test successful backend initialization."""

        mock_pynvml.nvmlInit.return_value = None
        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            assert backend.initialized
            mock_pynvml.nvmlInit.assert_called_once()

    def test_initialization_failure(self):
        """Test backend initialization failure."""

        mock_pynvml.nvmlInit.side_effect = Exception("NVML init failed")
        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            assert not backend.initialized

    def test_list_devices(self):
        """Test listing NVIDIA devices."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA A100"
        mock_pynvml.nvmlDeviceGetSerial.return_value = "1234567890"

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            devices = backend.list_devices()

            assert len(devices) == 2
            assert devices[0]["id"] == "GPU0"
            assert devices[0]["name"] == "NVIDIA A100"
            assert devices[1]["id"] == "GPU1"

    def test_get_state_full(self):
        """Test getting device state with all metrics available."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W in mW
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 300000  # 300W
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [1410, 1215]  # SM, MEM clocks
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 75
        mock_pynvml.nvmlDeviceGetEccMode.return_value = (1, 1)  # current, pending

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            state = backend.get_state("GPU0")

            assert state["power_mw"] == 250000
            assert state["power_limit_mw"] == 300000
            assert state["temp_c"] == 65
            assert state["sm_clock_mhz"] == 1410
            assert state["mem_clock_mhz"] == 1215
            assert state["fan_percent"] == 75
            assert state["ecc_enabled"] is True

    def test_get_state_partial(self):
        """Test getting device state when some metrics fail."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Not supported")
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            state = backend.get_state("GPU0")

            assert state["power_mw"] is None
            assert state["temp_c"] == 65

    def test_apply_setpoint_dry_run(self):
        """Test applying setpoint in dry-run mode."""

        mock_pynvml.nvmlInit.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=True)

            assert result["success"]
            assert result["dry_run"]
            assert result["device"] == "GPU0"
            assert result["parameter"] == "power_limit_w"
            assert result["value"] == 250

    def test_apply_setpoint_power_limit(self):
        """Test applying power limit setpoint."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=False)

            assert result["success"]
            assert not result["dry_run"]
            mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
                self.mock_handle, 250000
            )

    def test_apply_setpoint_sm_clock(self):
        """Test applying SM clock setpoint."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetApplicationsClocks.return_value = None
        # Mock current memory clock retrieval
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1215

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "sm_clock_mhz", 1410, dry_run=False)

            assert result["success"]
            # Should preserve current memory clock (1215) and set SM clock to 1410
            mock_pynvml.nvmlDeviceSetApplicationsClocks.assert_called_once_with(
                self.mock_handle, mem_clock=1215, sm_clock=1410
            )

    def test_apply_setpoint_mem_clock(self):
        """Test applying memory clock setpoint."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetApplicationsClocks.return_value = None
        # Mock current SM clock retrieval
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1410

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "mem_clock_mhz", 1215, dry_run=False)

            assert result["success"]
            # Should preserve current SM clock (1410) and set memory clock to 1215
            mock_pynvml.nvmlDeviceSetApplicationsClocks.assert_called_once_with(
                self.mock_handle, mem_clock=1215, sm_clock=1410
            )

    def test_apply_setpoint_fan_speed(self):
        """Test applying fan speed setpoint."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetFanSpeed_v2.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "fan_percent", 80, dry_run=False)

            assert result["success"]
            mock_pynvml.nvmlDeviceSetFanSpeed_v2.assert_called_once_with(self.mock_handle, 0, 80)

    def test_apply_setpoint_ecc_enabled(self):
        """Test enabling ECC."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetEccMode.return_value = None
        mock_pynvml.NVML_FEATURE_ENABLED = 1
        mock_pynvml.NVML_FEATURE_DISABLED = 0

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "ecc_enabled", True, dry_run=False)

            assert result["success"]
            mock_pynvml.nvmlDeviceSetEccMode.assert_called_once_with(self.mock_handle, 1)

    def test_apply_setpoint_ecc_disabled(self):
        """Test disabling ECC."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceSetEccMode.return_value = None
        mock_pynvml.NVML_FEATURE_ENABLED = 1
        mock_pynvml.NVML_FEATURE_DISABLED = 0

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "ecc_enabled", False, dry_run=False)

            assert result["success"]
            mock_pynvml.nvmlDeviceSetEccMode.assert_called_once_with(self.mock_handle, 0)

    def test_apply_setpoint_unknown_parameter(self):
        """Test applying unknown parameter returns error."""

        mock_pynvml.nvmlInit.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "unknown_param", 100, dry_run=False)

            assert not result["success"]
            assert "Unknown parameter" in result["error"]

    def test_apply_setpoint_exception(self):
        """Test apply setpoint handles exceptions."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Device error")

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.apply_setpoint("GPU0", "power_limit_w", 250, dry_run=False)

            assert not result["success"]
            assert "error" in result

    def test_reset_to_defaults_dry_run(self):
        """Test reset to defaults in dry-run mode."""

        mock_pynvml.nvmlInit.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.reset_to_defaults("GPU0", dry_run=True)

            assert result["success"]
            assert result["dry_run"]
            assert result["device"] == "GPU0"

    def test_reset_to_defaults(self):
        """Test reset to defaults."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceResetApplicationsClocks.return_value = None
        mock_pynvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.return_value = None

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.reset_to_defaults("GPU0", dry_run=False)

            assert result["success"]
            assert not result["dry_run"]
            mock_pynvml.nvmlDeviceResetApplicationsClocks.assert_called_once()

    def test_reset_to_defaults_partial_failure(self):
        """Test reset to defaults when power limit reset fails."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceResetApplicationsClocks.return_value = None
        mock_pynvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = Exception("Fail")

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.reset_to_defaults("GPU0", dry_run=False)

            # Should still succeed as power limit reset is optional
            assert result["success"]

    def test_reset_to_defaults_exception(self):
        """Test reset handles exceptions."""

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Device error")

        with patch("quasim.hardware.backends.nvml_backend.NVML_AVAILABLE", True):
            from quasim.hardware.backends.nvml_backend import NVMLBackend

            backend = NVMLBackend()
            result = backend.reset_to_defaults("GPU0", dry_run=False)

            assert not result["success"]
            assert "error" in result


if __name__ == "__main__":
    unittest.main()

"""

NVIDIA NVML backend for GPU reconfiguration.

Supports: clocks, power caps, MIG, ECC, fan control, app clocks.
Safe operations only - no firmware writes.
"""

import contextlib
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available - NVIDIA backend disabled")


class NVMLBackend:
    """NVIDIA GPU control via NVML."""

    def __init__(self):
        self.initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.initialized = True
                logger.info("NVML backend initialized")
            except Exception as e:
                logger.error(f"NVML initialization failed: {e}")

    def __del__(self):
        if self.initialized and NVML_AVAILABLE:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    def list_devices(self) -> List[Dict[str, Any]]:
        """List available NVIDIA GPUs."""

        if not self.initialized:
            return []

        devices = []
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                serial = pynvml.nvmlDeviceGetSerial(handle)

                # Handle both str and bytes returns from pynvml
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                if isinstance(serial, bytes):
                    serial = serial.decode("utf-8")

                devices.append(
                    {
                        "id": f"GPU{i}",
                        "index": i,
                        "name": name,
                        "serial": serial,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to list NVIDIA devices: {e}")

        return devices

    def get_state(self, device_id: str) -> Dict[str, Any]:
        """Get current device state."""

        if not self.initialized:
            return {}

        try:
            idx = int(device_id.replace("GPU", ""))
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            # Power
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            except Exception:
                power_mw = power_limit_mw = None

            # Temperature
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp_c = None

            # Clocks
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                sm_clock = mem_clock = None

            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                fan_speed = None

            # ECC mode
            try:
                ecc_current, _ecc_pending = pynvml.nvmlDeviceGetEccMode(handle)
            except Exception:
                ecc_current = None

            return {
                "power_mw": power_mw,
                "power_limit_mw": power_limit_mw,
                "temp_c": temp_c,
                "sm_clock_mhz": sm_clock,
                "mem_clock_mhz": mem_clock,
                "fan_percent": fan_speed,
                "ecc_enabled": (
                    (ecc_current == pynvml.NVML_FEATURE_ENABLED)
                    if ecc_current is not None
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get state for {device_id}: {e}")
            return {}

    def apply_setpoint(
        self, device_id: str, parameter: str, value: Any, dry_run: bool = True
    ) -> Dict[str, Any]:
        """

        Apply setpoint to device.

        Supported parameters:
        - power_limit_w: Power cap in watts
        - sm_clock_mhz: SM clock frequency
        - mem_clock_mhz: Memory clock frequency
        - fan_percent: Fan speed percentage
        - ecc_enabled: Enable/disable ECC
        """

        if not self.initialized:
            return {"success": False, "error": "NVML not initialized"}

        try:
            idx = int(device_id.replace("GPU", ""))
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            if dry_run:
                logger.info(f"[DRY-RUN] Would set {device_id}.{parameter} = {value}")
                return {
                    "success": True,
                    "dry_run": True,
                    "device": device_id,
                    "parameter": parameter,
                    "value": value,
                }

            # Apply actual setpoint
            if parameter == "power_limit_w":
                power_limit_mw = int(value * 1000)
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit_mw)
                logger.info(f"Set {device_id} power limit to {value}W")

            elif parameter == "sm_clock_mhz":
                # Get current memory clock to preserve it
                try:
                    current_mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    current_mem_clock = 0
                pynvml.nvmlDeviceSetApplicationsClocks(
                    handle, mem_clock=current_mem_clock, sm_clock=int(value)
                )
                logger.info(f"Set {device_id} SM clock to {value}MHz")

            elif parameter == "mem_clock_mhz":
                # Get current SM clock to preserve it
                try:
                    current_sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                except Exception:
                    current_sm_clock = 0
                pynvml.nvmlDeviceSetApplicationsClocks(
                    handle, mem_clock=int(value), sm_clock=current_sm_clock
                )
                logger.info(f"Set {device_id} memory clock to {value}MHz")

            elif parameter == "fan_percent":
                pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, int(value))
                logger.info(f"Set {device_id} fan speed to {value}%")

            elif parameter == "ecc_enabled":
                mode = pynvml.NVML_FEATURE_ENABLED if value else pynvml.NVML_FEATURE_DISABLED
                pynvml.nvmlDeviceSetEccMode(handle, mode)
                logger.info(f"Set {device_id} ECC to {'enabled' if value else 'disabled'}")

            else:
                return {"success": False, "error": f"Unknown parameter: {parameter}"}

            return {
                "success": True,
                "dry_run": False,
                "device": device_id,
                "parameter": parameter,
                "value": value,
            }

        except Exception as e:
            logger.error(f"Failed to apply setpoint: {e}")
            return {"success": False, "error": str(e)}

    def reset_to_defaults(self, device_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """Reset device to default/safe configuration."""

        if not self.initialized:
            return {"success": False, "error": "NVML not initialized"}

        try:
            idx = int(device_id.replace("GPU", ""))
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

            if dry_run:
                logger.info(f"[DRY-RUN] Would reset {device_id} to defaults")
                return {"success": True, "dry_run": True, "device": device_id}

            # Reset to default clocks
            pynvml.nvmlDeviceResetApplicationsClocks(handle)

            # Reset power limit to default
            try:
                default_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_limit)
            except Exception as e:
                logger.warning(f"Failed to reset power management limit for {device_id}: {e}")

            logger.info(f"Reset {device_id} to defaults")
            return {"success": True, "dry_run": False, "device": device_id}

        except Exception as e:
            logger.error(f"Failed to reset {device_id}: {e}")
            return {"success": False, "error": str(e)}

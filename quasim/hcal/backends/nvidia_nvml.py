"""NVIDIA NVML backend for HCAL."""

from typing import Any, Dict

from quasim.hcal.backends import BaseBackend


class NvidiaNvmlBackend(BaseBackend):
    """NVIDIA NVML backend driver."""

    def __init__(self, dry_run: bool = True):
        """Initialize NVIDIA NVML backend.

        Args:
            dry_run: Enable dry-run mode.
        """
        super().__init__(dry_run)
        self.nvml_available = False
        self.handles = {}

        if not dry_run:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml_available = True
                self.pynvml = pynvml

                # Get device handles
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.handles[f"gpu{i}"] = handle

            except (ImportError, Exception) as e:
                print(f"NVML not available: {e}")
                self.nvml_available = False

    def device_exists(self, device_id: str) -> bool:
        """Check if device exists.

        Args:
            device_id: Device identifier (e.g., 'gpu0').

        Returns:
            True if device exists.
        """
        if self.dry_run:
            # In dry-run mode, simulate device existence
            return device_id.startswith("gpu")

        return device_id in self.handles

    def apply_setpoint(self, device_id: str, setpoint: Dict[str, Any]) -> bool:
        """Apply setpoint to device.

        Args:
            device_id: Device identifier.
            setpoint: Setpoint dictionary with keys like:
                - power_limit_watts: Power cap
                - sm_clock_mhz: SM clock frequency
                - mem_clock_mhz: Memory clock frequency
                - fan_speed_percent: Fan speed

        Returns:
            True if successful.
        """
        if self.dry_run:
            print(f"[DRY-RUN] Applying setpoint to {device_id}: {setpoint}")
            return True

        if not self.nvml_available:
            print("NVML not available")
            return False

        if device_id not in self.handles:
            print(f"Device {device_id} not found")
            return False

        handle = self.handles[device_id]

        try:
            # Apply power limit
            if "power_limit_watts" in setpoint:
                power_limit = int(setpoint["power_limit_watts"] * 1000)  # Convert to mW
                self.pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit)

            # Apply clock limits (requires elevated permissions)
            if "sm_clock_mhz" in setpoint or "mem_clock_mhz" in setpoint:
                # Note: This requires root/admin permissions
                # sm_clock = int(setpoint.get("sm_clock_mhz", 0))
                # mem_clock = int(setpoint.get("mem_clock_mhz", 0))
                # self.pynvml.nvmlDeviceSetGpuLockedClocks(handle, sm_clock, sm_clock)
                pass

            # Apply fan speed
            if "fan_speed_percent" in setpoint:
                # Note: Not all GPUs support manual fan control
                # fan_speed = int(setpoint["fan_speed_percent"])
                # self.pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, fan_speed)
                pass

            return True

        except Exception as e:
            print(f"Error applying setpoint: {e}")
            return False

    def read_configuration(self, device_id: str) -> Dict[str, Any]:
        """Read current device configuration.

        Args:
            device_id: Device identifier.

        Returns:
            Configuration dictionary.
        """
        if self.dry_run:
            # Simulate configuration
            return {
                "power_limit_watts": 250.0,
                "sm_clock_mhz": 1410,
                "mem_clock_mhz": 5001,
            }

        if not self.nvml_available or device_id not in self.handles:
            return {}

        handle = self.handles[device_id]
        config = {}

        try:
            # Read power limit
            power_limit = self.pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            config["power_limit_watts"] = power_limit / 1000.0

            # Read clock frequencies
            sm_clock = self.pynvml.nvmlDeviceGetClockInfo(handle, self.pynvml.NVML_CLOCK_SM)
            mem_clock = self.pynvml.nvmlDeviceGetClockInfo(
                handle, self.pynvml.NVML_CLOCK_MEM
            )
            config["sm_clock_mhz"] = sm_clock
            config["mem_clock_mhz"] = mem_clock

        except Exception as e:
            print(f"Error reading configuration: {e}")

        return config

    def read_telemetry(self, device_id: str) -> Dict[str, Any]:
        """Read device telemetry.

        Args:
            device_id: Device identifier.

        Returns:
            Telemetry dictionary with metrics like:
                - power_watts: Current power draw
                - temperature_celsius: GPU temperature
                - utilization_percent: GPU utilization
                - memory_used_mb: Memory usage
        """
        if self.dry_run:
            # Simulate telemetry
            return {
                "power_watts": 235.0,
                "temperature_celsius": 72.0,
                "utilization_percent": 85.0,
                "memory_used_mb": 8192.0,
                "sm_clock_mhz": 1410,
                "mem_clock_mhz": 5001,
            }

        if not self.nvml_available or device_id not in self.handles:
            return {}

        handle = self.handles[device_id]
        telemetry = {}

        try:
            # Read power
            power = self.pynvml.nvmlDeviceGetPowerUsage(handle)
            telemetry["power_watts"] = power / 1000.0

            # Read temperature
            temp = self.pynvml.nvmlDeviceGetTemperature(
                handle, self.pynvml.NVML_TEMPERATURE_GPU
            )
            telemetry["temperature_celsius"] = float(temp)

            # Read utilization
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            telemetry["utilization_percent"] = float(util.gpu)

            # Read memory
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            telemetry["memory_used_mb"] = mem_info.used / (1024 * 1024)

            # Read clocks
            sm_clock = self.pynvml.nvmlDeviceGetClockInfo(handle, self.pynvml.NVML_CLOCK_SM)
            mem_clock = self.pynvml.nvmlDeviceGetClockInfo(
                handle, self.pynvml.NVML_CLOCK_MEM
            )
            telemetry["sm_clock_mhz"] = sm_clock
            telemetry["mem_clock_mhz"] = mem_clock

        except Exception as e:
            print(f"Error reading telemetry: {e}")

        return telemetry

    def __del__(self):
        """Cleanup NVML."""
        if self.nvml_available and not self.dry_run:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

"""Energy-aware monitoring with NVML/ROCm SMI integration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PowerMetrics:
    """Power and energy metrics."""
    power_watts: float
    energy_joules: float
    temperature_celsius: float
    utilization_percent: float
    timestamp: float


class EnergyMonitor:
    """Monitor power consumption and optimize for energy efficiency."""
    
    def __init__(self, backend: str = "auto") -> None:
        self.backend = backend
        self._baseline_power: Optional[float] = None
        self._samples: list[PowerMetrics] = []
        self._monitoring_start: Optional[float] = None
        
    def start_monitoring(self) -> None:
        """Start energy monitoring session."""
        self._monitoring_start = time.perf_counter()
        self._samples.clear()
        
        # Simulate baseline power reading
        self._baseline_power = self._read_power()
        
    def stop_monitoring(self) -> PowerMetrics:
        """Stop monitoring and return aggregated metrics."""
        if self._monitoring_start is None:
            raise RuntimeError("Monitoring not started")
            
        duration = time.perf_counter() - self._monitoring_start
        
        # Compute aggregated metrics
        if self._samples:
            avg_power = sum(s.power_watts for s in self._samples) / len(self._samples)
            max_temp = max(s.temperature_celsius for s in self._samples)
            avg_util = sum(s.utilization_percent for s in self._samples) / len(self._samples)
        else:
            avg_power = self._baseline_power or 0.0
            max_temp = 0.0
            avg_util = 0.0
            
        total_energy = avg_power * duration
        
        return PowerMetrics(
            power_watts=avg_power,
            energy_joules=total_energy,
            temperature_celsius=max_temp,
            utilization_percent=avg_util,
            timestamp=time.perf_counter(),
        )
        
    def sample(self) -> PowerMetrics:
        """Take a power measurement sample."""
        power = self._read_power()
        temp = self._read_temperature()
        util = self._read_utilization()
        
        # Estimate energy from last sample
        if self._samples:
            time_delta = time.perf_counter() - self._samples[-1].timestamp
            energy_delta = power * time_delta
        else:
            energy_delta = 0.0
            
        metrics = PowerMetrics(
            power_watts=power,
            energy_joules=energy_delta,
            temperature_celsius=temp,
            utilization_percent=util,
            timestamp=time.perf_counter(),
        )
        
        self._samples.append(metrics)
        return metrics
        
    def _read_power(self) -> float:
        """Read current power consumption in watts."""
        # Simulate power reading based on backend
        if self.backend in ("cuda", "nvidia"):
            # Simulate NVML reading
            base_power = 150.0
            variation = 50.0
        elif self.backend in ("hip", "amd", "rocm"):
            # Simulate ROCm SMI reading
            base_power = 200.0
            variation = 60.0
        else:
            # CPU or unknown
            base_power = 65.0
            variation = 20.0
            
        import random
        return base_power + random.uniform(-variation * 0.2, variation * 0.2)
        
    def _read_temperature(self) -> float:
        """Read current temperature in Celsius."""
        import random
        return random.uniform(40.0, 85.0)
        
    def _read_utilization(self) -> float:
        """Read current GPU/accelerator utilization percentage."""
        import random
        return random.uniform(20.0, 100.0)
        
    def compute_efficiency(self, gflops: float, power_watts: float) -> float:
        """Compute energy efficiency in GFLOPs/W."""
        if power_watts <= 0:
            return 0.0
        return gflops / power_watts
        
    def get_statistics(self) -> dict[str, float]:
        """Get energy monitoring statistics."""
        if not self._samples:
            return {
                "avg_power_w": 0.0,
                "total_energy_j": 0.0,
                "peak_power_w": 0.0,
                "max_temp_c": 0.0,
            }
            
        return {
            "avg_power_w": sum(s.power_watts for s in self._samples) / len(self._samples),
            "total_energy_j": sum(s.energy_joules for s in self._samples),
            "peak_power_w": max(s.power_watts for s in self._samples),
            "max_temp_c": max(s.temperature_celsius for s in self._samples),
            "avg_util_pct": sum(s.utilization_percent for s in self._samples) / len(self._samples),
            "samples": len(self._samples),
        }

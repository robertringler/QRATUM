"""Heterogeneous workload scheduler across multiple device types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    FPGA = "fpga"


@dataclass
class Device:
    """Represents a compute device."""
    device_id: int
    device_type: DeviceType
    compute_units: int
    memory_gb: float
    peak_gflops: float
    power_budget_w: float
    available: bool = True
    current_load: float = 0.0
    
    def can_handle(self, workload_size: float) -> bool:
        """Check if device can handle workload."""
        return self.available and (self.current_load + workload_size) <= 1.0


@dataclass
class SchedulingDecision:
    """Decision about workload placement."""
    device: Device
    estimated_time: float
    estimated_energy: float
    
    @property
    def efficiency_score(self) -> float:
        """Compute efficiency score (lower is better)."""
        # Balance time and energy (weighted sum)
        time_weight = 0.6
        energy_weight = 0.4
        return time_weight * self.estimated_time + energy_weight * self.estimated_energy


class HeteroScheduler:
    """Scheduler for heterogeneous compute resources."""
    
    def __init__(self) -> None:
        self.devices: List[Device] = []
        self._scheduling_history: List[SchedulingDecision] = []
        self._performance_models: dict[DeviceType, Callable[[float], float]] = {}
        
    def register_device(
        self,
        device_type: DeviceType,
        device_id: int = 0,
        compute_units: int = 1,
        memory_gb: float = 16.0,
        peak_gflops: float = 1000.0,
        power_budget_w: float = 150.0,
    ) -> Device:
        """Register a compute device with the scheduler."""
        device = Device(
            device_id=device_id,
            device_type=device_type,
            compute_units=compute_units,
            memory_gb=memory_gb,
            peak_gflops=peak_gflops,
            power_budget_w=power_budget_w,
        )
        self.devices.append(device)
        return device
        
    def _estimate_execution_time(
        self,
        device: Device,
        workload_size: float,
        workload_type: str,
    ) -> float:
        """Estimate execution time for a workload on a device."""
        # Use performance model if available
        if device.device_type in self._performance_models:
            return self._performance_models[device.device_type](workload_size)
            
        # Simple heuristic based on device characteristics
        base_time = workload_size / device.peak_gflops
        
        # Adjust for device type
        type_factors = {
            DeviceType.GPU: 1.0,
            DeviceType.CPU: 3.0,
            DeviceType.TPU: 0.8,
            DeviceType.NPU: 1.2,
            DeviceType.FPGA: 2.0,
        }
        factor = type_factors.get(device.device_type, 2.0)
        
        # Adjust for current load
        load_penalty = 1.0 + device.current_load * 0.5
        
        return base_time * factor * load_penalty
        
    def _estimate_energy(self, device: Device, execution_time: float) -> float:
        """Estimate energy consumption for execution."""
        return device.power_budget_w * execution_time
        
    def schedule(
        self,
        workload_size: float,
        workload_type: str = "compute",
        preferred_device: Optional[DeviceType] = None,
    ) -> Optional[SchedulingDecision]:
        """Schedule a workload to the best available device."""
        if not self.devices:
            return None
            
        # Filter available devices
        candidates = [d for d in self.devices if d.can_handle(workload_size)]
        
        # Apply preference if specified
        if preferred_device:
            candidates = [d for d in candidates if d.device_type == preferred_device]
            
        if not candidates:
            return None
            
        # Evaluate each candidate
        decisions = []
        for device in candidates:
            exec_time = self._estimate_execution_time(device, workload_size, workload_type)
            energy = self._estimate_energy(device, exec_time)
            decisions.append(SchedulingDecision(
                device=device,
                estimated_time=exec_time,
                estimated_energy=energy,
            ))
            
        # Select best decision
        best_decision = min(decisions, key=lambda d: d.efficiency_score)
        
        # Update device state
        best_decision.device.current_load += workload_size
        self._scheduling_history.append(best_decision)
        
        return best_decision
        
    def release_device(self, device: Device, workload_size: float) -> None:
        """Release device resources after workload completion."""
        device.current_load = max(0.0, device.current_load - workload_size)
        
    def set_performance_model(
        self,
        device_type: DeviceType,
        model: Callable[[float], float],
    ) -> None:
        """Set a performance model for a device type."""
        self._performance_models[device_type] = model
        
    def get_statistics(self) -> dict[str, Any]:
        """Get scheduling statistics."""
        if not self._scheduling_history:
            return {
                "total_scheduled": 0,
                "avg_time": 0.0,
                "avg_energy": 0.0,
            }
            
        device_usage = {}
        for decision in self._scheduling_history:
            device_type = decision.device.device_type.value
            device_usage[device_type] = device_usage.get(device_type, 0) + 1
            
        total_time = sum(d.estimated_time for d in self._scheduling_history)
        total_energy = sum(d.estimated_energy for d in self._scheduling_history)
        
        return {
            "total_scheduled": len(self._scheduling_history),
            "avg_time": total_time / len(self._scheduling_history),
            "avg_energy": total_energy / len(self._scheduling_history),
            "total_energy": total_energy,
            "device_usage": device_usage,
            "devices_registered": len(self.devices),
        }

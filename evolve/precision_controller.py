"""Hierarchical hybrid precision graph controller for Phase III.

Implements multi-level precision zoning (outer FP32 → inner FP8/INT4 → boundary BF16)
with automatic fallback when accumulated error exceeds threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    
    @property
    def bits(self) -> int:
        """Return number of bits for this precision."""
        bit_map = {
            "fp32": 32, "fp16": 16, "bf16": 16,
            "fp8": 8, "int8": 8, "int4": 4
        }
        return bit_map[self.value]
    
    @property
    def error_tolerance(self) -> float:
        """Typical numerical error tolerance for this precision."""
        tolerance_map = {
            "fp32": 1e-7, "fp16": 1e-3, "bf16": 1e-2,
            "fp8": 1e-2, "int8": 1e-1, "int4": 1e-1
        }
        return tolerance_map[self.value]


@dataclass
class PrecisionZone:
    """Represents a precision zone in the computation graph."""
    
    zone_id: str
    zone_type: str  # "outer", "inner", "boundary"
    precision: PrecisionType
    layers: List[str] = field(default_factory=list)
    accumulated_error: float = 0.0
    compute_savings_pct: float = 0.0


@dataclass
class PrecisionMap:
    """Complete precision mapping for a kernel."""
    
    kernel_id: str
    zones: List[PrecisionZone]
    global_error_budget: float = 1e-5
    total_error: float = 0.0
    fallback_active: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kernel_id": self.kernel_id,
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "zone_type": z.zone_type,
                    "precision": z.precision.value,
                    "layers": z.layers,
                    "accumulated_error": z.accumulated_error,
                    "compute_savings_pct": z.compute_savings_pct
                }
                for z in self.zones
            ],
            "global_error_budget": self.global_error_budget,
            "total_error": self.total_error,
            "fallback_active": self.fallback_active
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> PrecisionMap:
        """Create from dictionary."""
        zones = [
            PrecisionZone(
                zone_id=z["zone_id"],
                zone_type=z["zone_type"],
                precision=PrecisionType(z["precision"]),
                layers=z["layers"],
                accumulated_error=z["accumulated_error"],
                compute_savings_pct=z.get("compute_savings_pct", 0.0)
            )
            for z in data["zones"]
        ]
        return cls(
            kernel_id=data["kernel_id"],
            zones=zones,
            global_error_budget=data["global_error_budget"],
            total_error=data["total_error"],
            fallback_active=data.get("fallback_active", False)
        )


class PrecisionController:
    """Controller for hierarchical precision management."""
    
    def __init__(self, maps_dir: Optional[Path] = None):
        """Initialize precision controller.
        
        Args:
            maps_dir: Directory to store precision maps
        """
        self.maps_dir = maps_dir or Path(__file__).parent.parent / "schedules" / "precision_maps"
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self._active_maps: Dict[str, PrecisionMap] = {}
    
    def create_hierarchical_map(self, kernel_id: str, num_layers: int = 10) -> PrecisionMap:
        """Create a hierarchical precision map for a kernel.
        
        Args:
            kernel_id: Identifier for the kernel
            num_layers: Number of layers in the computation
            
        Returns:
            PrecisionMap with hierarchical zones
        """
        zones = []
        
        # Outer zone: FP32 for stability
        outer_layers = [f"layer_{i}" for i in range(2)]
        zones.append(PrecisionZone(
            zone_id=f"{kernel_id}_outer",
            zone_type="outer",
            precision=PrecisionType.FP32,
            layers=outer_layers,
            compute_savings_pct=0.0
        ))
        
        # Inner zone: FP8 for performance
        inner_layers = [f"layer_{i}" for i in range(2, num_layers - 2)]
        zones.append(PrecisionZone(
            zone_id=f"{kernel_id}_inner",
            zone_type="inner",
            precision=PrecisionType.FP8,
            layers=inner_layers,
            compute_savings_pct=75.0  # 8-bit vs 32-bit
        ))
        
        # Boundary zone: BF16 for balance
        boundary_layers = [f"layer_{i}" for i in range(num_layers - 2, num_layers)]
        zones.append(PrecisionZone(
            zone_id=f"{kernel_id}_boundary",
            zone_type="boundary",
            precision=PrecisionType.BF16,
            layers=boundary_layers,
            compute_savings_pct=50.0
        ))
        
        precision_map = PrecisionMap(
            kernel_id=kernel_id,
            zones=zones,
            global_error_budget=1e-5
        )
        
        self._active_maps[kernel_id] = precision_map
        return precision_map
    
    def update_error_estimate(self, kernel_id: str, zone_id: str, 
                             error_delta: float) -> None:
        """Update accumulated error for a zone.
        
        Args:
            kernel_id: Kernel identifier
            zone_id: Zone identifier
            error_delta: Error to add to accumulated error
        """
        if kernel_id not in self._active_maps:
            raise ValueError(f"No precision map for kernel: {kernel_id}")
        
        precision_map = self._active_maps[kernel_id]
        
        for zone in precision_map.zones:
            if zone.zone_id == zone_id:
                zone.accumulated_error += error_delta
                break
        
        # Update total error
        precision_map.total_error = sum(z.accumulated_error for z in precision_map.zones)
        
        # Check if fallback needed
        if precision_map.total_error > precision_map.global_error_budget:
            self.activate_fallback(kernel_id)
    
    def activate_fallback(self, kernel_id: str) -> None:
        """Activate mixed-precision fallback for a kernel.
        
        Upgrades all zones to higher precision when error budget exceeded.
        
        Args:
            kernel_id: Kernel identifier
        """
        if kernel_id not in self._active_maps:
            return
        
        precision_map = self._active_maps[kernel_id]
        precision_map.fallback_active = True
        
        # Upgrade precision for all zones
        for zone in precision_map.zones:
            if zone.precision == PrecisionType.FP8:
                zone.precision = PrecisionType.FP16
                zone.compute_savings_pct = 50.0
            elif zone.precision == PrecisionType.INT4:
                zone.precision = PrecisionType.INT8
                zone.compute_savings_pct = 75.0
        
        # Reset error accumulation
        for zone in precision_map.zones:
            zone.accumulated_error = 0.0
        precision_map.total_error = 0.0
    
    def save_map(self, kernel_id: str) -> Path:
        """Save precision map to disk.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Path to saved map
        """
        if kernel_id not in self._active_maps:
            raise ValueError(f"No precision map for kernel: {kernel_id}")
        
        map_path = self.maps_dir / f"{kernel_id}_precision_map.json"
        precision_map = self._active_maps[kernel_id]
        
        with open(map_path, 'w') as f:
            json.dump(precision_map.to_dict(), f, indent=2)
        
        return map_path
    
    def load_map(self, kernel_id: str) -> Optional[PrecisionMap]:
        """Load precision map from disk.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Loaded PrecisionMap or None if not found
        """
        map_path = self.maps_dir / f"{kernel_id}_precision_map.json"
        if not map_path.exists():
            return None
        
        with open(map_path, 'r') as f:
            data = json.load(f)
        
        precision_map = PrecisionMap.from_dict(data)
        self._active_maps[kernel_id] = precision_map
        return precision_map
    
    def get_compute_savings(self, kernel_id: str) -> float:
        """Calculate total compute savings from precision optimization.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Percentage of compute savings
        """
        if kernel_id not in self._active_maps:
            return 0.0
        
        precision_map = self._active_maps[kernel_id]
        total_layers = sum(len(z.layers) for z in precision_map.zones)
        
        if total_layers == 0:
            return 0.0
        
        weighted_savings = sum(
            (len(z.layers) / total_layers) * z.compute_savings_pct
            for z in precision_map.zones
        )
        
        return weighted_savings


if __name__ == "__main__":
    print("Hierarchical Precision Controller Demo")
    print("=" * 60)
    
    # Initialize controller
    controller = PrecisionController()
    
    # Create precision map
    kernel_id = "matmul_kernel_001"
    precision_map = controller.create_hierarchical_map(kernel_id, num_layers=10)
    
    print(f"\nPrecision Map for {kernel_id}:")
    for zone in precision_map.zones:
        print(f"\n  Zone: {zone.zone_id}")
        print(f"    Type:      {zone.zone_type}")
        print(f"    Precision: {zone.precision.value} ({zone.precision.bits} bits)")
        print(f"    Layers:    {len(zone.layers)}")
        print(f"    Savings:   {zone.compute_savings_pct:.1f}%")
    
    # Simulate error accumulation
    print("\n" + "=" * 60)
    print("Simulating error accumulation...")
    
    controller.update_error_estimate(kernel_id, f"{kernel_id}_inner", 3e-6)
    controller.update_error_estimate(kernel_id, f"{kernel_id}_boundary", 5e-6)
    
    print(f"Total Error: {precision_map.total_error:.2e}")
    print(f"Error Budget: {precision_map.global_error_budget:.2e}")
    
    if precision_map.total_error > precision_map.global_error_budget:
        print("\n⚠️  Error budget exceeded, activating fallback...")
        controller.activate_fallback(kernel_id)
        print("✓ Fallback activated, precision upgraded")
    
    # Calculate savings
    savings = controller.get_compute_savings(kernel_id)
    print(f"\nTotal Compute Savings: {savings:.1f}%")
    
    # Save map
    map_path = controller.save_map(kernel_id)
    print(f"Precision map saved to: {map_path}")

"""Hierarchical hybrid precision management system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PrecisionLevel(Enum):
    """Supported precision levels."""

    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT4 = "int4"
    INT8 = "int8"


@dataclass
class PrecisionZone:
    """Defines a precision zone within a kernel."""

    zone_id: str
    precision: PrecisionLevel
    region: str  # "outer", "inner", "boundary"
    error_tolerance: float = 1e-5

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "zone_id": self.zone_id,
            "precision": self.precision.value,
            "region": self.region,
            "error_tolerance": self.error_tolerance,
        }


@dataclass
class PrecisionMap:
    """Precision mapping for a kernel."""

    kernel_id: str
    zones: list[PrecisionZone]
    global_error_budget: float = 1e-5
    accumulated_error: float = 0.0
    fallback_precision: PrecisionLevel = PrecisionLevel.FP32

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "kernel_id": self.kernel_id,
            "zones": [z.to_dict() for z in self.zones],
            "global_error_budget": self.global_error_budget,
            "accumulated_error": self.accumulated_error,
            "fallback_precision": self.fallback_precision.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PrecisionMap:
        """Create from dictionary."""

        zones = [
            PrecisionZone(
                zone_id=z["zone_id"],
                precision=PrecisionLevel(z["precision"]),
                region=z["region"],
                error_tolerance=z["error_tolerance"],
            )
            for z in data["zones"]
        ]
        return cls(
            kernel_id=data["kernel_id"],
            zones=zones,
            global_error_budget=data["global_error_budget"],
            accumulated_error=data["accumulated_error"],
            fallback_precision=PrecisionLevel(data["fallback_precision"]),
        )


class PrecisionManager:
    """Manages hierarchical hybrid precision for kernels."""

    def __init__(self, map_dir: str = "evolve/precision_maps"):
        self.map_dir = Path(map_dir)
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.precision_maps: dict[str, PrecisionMap] = {}

    def create_default_map(self, kernel_id: str) -> PrecisionMap:
        """Create default hierarchical precision map for a kernel."""

        zones = [
            PrecisionZone(
                zone_id="outer",
                precision=PrecisionLevel.FP32,
                region="outer",
                error_tolerance=1e-6,
            ),
            PrecisionZone(
                zone_id="inner",
                precision=PrecisionLevel.FP8,
                region="inner",
                error_tolerance=1e-4,
            ),
            PrecisionZone(
                zone_id="boundary",
                precision=PrecisionLevel.BF16,
                region="boundary",
                error_tolerance=1e-5,
            ),
        ]

        precision_map = PrecisionMap(
            kernel_id=kernel_id,
            zones=zones,
            global_error_budget=1e-5,
        )

        self.precision_maps[kernel_id] = precision_map
        return precision_map

    def update_accumulated_error(self, kernel_id: str, error: float) -> bool:
        """

        Update accumulated error for a kernel.
        Returns True if fallback is needed.
        """

        if kernel_id not in self.precision_maps:
            self.create_default_map(kernel_id)

        precision_map = self.precision_maps[kernel_id]
        precision_map.accumulated_error += error

        # Check if we need fallback
        needs_fallback = precision_map.accumulated_error > precision_map.global_error_budget

        if needs_fallback:
            # Trigger automatic mixed-precision fallback
            self._apply_fallback(precision_map)

        return needs_fallback

    def _apply_fallback(self, precision_map: PrecisionMap) -> None:
        """Apply fallback to higher precision."""

        print(
            f"Warning: Accumulated error {precision_map.accumulated_error:.2e} "
            f"exceeds budget {precision_map.global_error_budget:.2e}"
        )
        print(f"Applying fallback to {precision_map.fallback_precision.value}")

        # Upgrade all zones to fallback precision
        for zone in precision_map.zones:
            zone.precision = precision_map.fallback_precision

        # Reset accumulated error
        precision_map.accumulated_error = 0.0

    def save_map(self, kernel_id: str) -> Path:
        """Save precision map to disk."""

        if kernel_id not in self.precision_maps:
            raise ValueError(f"No precision map for kernel {kernel_id}")

        precision_map = self.precision_maps[kernel_id]
        output_path = self.map_dir / f"{kernel_id}_precision.json"

        with open(output_path, "w") as f:
            json.dump(precision_map.to_dict(), f, indent=2)

        return output_path

    def load_map(self, kernel_id: str) -> PrecisionMap:
        """Load precision map from disk."""

        map_path = self.map_dir / f"{kernel_id}_precision.json"

        if not map_path.exists():
            return self.create_default_map(kernel_id)

        with open(map_path) as f:
            data = json.load(f)

        precision_map = PrecisionMap.from_dict(data)
        self.precision_maps[kernel_id] = precision_map

        return precision_map

    def get_precision_for_zone(self, kernel_id: str, zone_id: str) -> PrecisionLevel:
        """Get precision level for a specific zone."""

        if kernel_id not in self.precision_maps:
            self.create_default_map(kernel_id)

        precision_map = self.precision_maps[kernel_id]

        for zone in precision_map.zones:
            if zone.zone_id == zone_id:
                return zone.precision

        # Default to FP32 if zone not found
        return PrecisionLevel.FP32

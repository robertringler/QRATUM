"""Tire geometry and structural design modeling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TireType(Enum):
    """Types of tires."""

    PASSENGER = "passenger"
    TRUCK = "truck"
    OFF_ROAD = "off_road"
    RACING = "racing"
    EV_SPECIFIC = "ev_specific"
    WINTER = "winter"
    ALL_SEASON = "all_season"
    PERFORMANCE = "performance"


class TreadPattern(Enum):
    """Tread pattern types."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DIRECTIONAL = "directional"
    MULTIDIRECTIONAL = "multidirectional"


@dataclass
class TreadDesign:
    """Tread geometry and surface pattern specification.

    Attributes:
        pattern_type: Overall tread pattern type
        groove_depth: Main groove depth in mm
        groove_width: Main groove width in mm
        groove_count: Number of main grooves
        sipe_density: Sipes per cm²
        sipe_depth: Sipe depth in mm
        microtexture_roughness: Surface roughness in μm
        void_ratio: Percentage of open area (0-1)
        edge_count: Number of biting edges
        block_stiffness: Tread block stiffness index (0-1)
    """

    pattern_type: TreadPattern = TreadPattern.SYMMETRIC
    groove_depth: float = 8.0  # mm
    groove_width: float = 8.0  # mm
    groove_count: int = 4
    sipe_density: float = 5.0  # sipes/cm²
    sipe_depth: float = 4.0  # mm
    microtexture_roughness: float = 50.0  # μm
    void_ratio: float = 0.3
    edge_count: int = 200
    block_stiffness: float = 0.7

    def compute_water_evacuation_capacity(self, speed_kmh: float) -> float:
        """Compute water evacuation capacity for hydroplaning resistance.

        Args:
            speed_kmh: Vehicle speed in km/h

        Returns:
            Water evacuation capacity in L/s
        """
        # Base capacity from groove volume
        groove_volume = self.groove_depth * self.groove_width * self.groove_count
        base_capacity = groove_volume * 0.001  # Convert to L/s

        # Speed effect
        speed_factor = 1.0 + 0.01 * speed_kmh

        # Void ratio contribution
        void_factor = 1.0 + 2.0 * self.void_ratio

        return base_capacity * speed_factor * void_factor

    def compute_traction_index(self, surface_wetness: float) -> float:
        """Compute traction index for given surface conditions.

        Args:
            surface_wetness: Surface wetness level (0=dry, 1=fully wet)

        Returns:
            Traction index (0-1)
        """
        # Dry traction primarily from block stiffness and edges
        dry_traction = 0.4 * self.block_stiffness + 0.3 * (self.edge_count / 300.0)

        # Wet traction from sipes and microtexture
        wet_traction = 0.3 * (self.sipe_density / 10.0) + 0.2 * (
            self.microtexture_roughness / 100.0
        )

        # Interpolate based on wetness
        return (1.0 - surface_wetness) * dry_traction + surface_wetness * wet_traction

    def to_dict(self) -> dict[str, Any]:
        """Serialize tread design to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "groove_depth": self.groove_depth,
            "groove_width": self.groove_width,
            "groove_count": self.groove_count,
            "sipe_density": self.sipe_density,
            "void_ratio": self.void_ratio,
            "edge_count": self.edge_count,
        }


@dataclass
class TireStructure:
    """Internal tire structure specification.

    Attributes:
        belt_count: Number of belt layers
        ply_count: Number of ply layers
        belt_angle: Belt angle in degrees
        carcass_type: Carcass construction type (radial/bias)
        bead_type: Bead construction type
        sidewall_thickness: Sidewall thickness in mm
        air_volume: Internal air volume in liters
        max_load: Maximum load rating in kg
        max_pressure: Maximum pressure in kPa
    """

    belt_count: int = 2
    ply_count: int = 2
    belt_angle: float = 22.0  # degrees
    carcass_type: str = "radial"
    bead_type: str = "steel"
    sidewall_thickness: float = 8.0  # mm
    air_volume: float = 25.0  # liters
    max_load: float = 800.0  # kg
    max_pressure: float = 300.0  # kPa

    def compute_load_capacity(self, pressure_kpa: float, temperature_c: float) -> float:
        """Compute load capacity at given pressure and temperature.

        Args:
            pressure_kpa: Inflation pressure in kPa
            temperature_c: Temperature in °C

        Returns:
            Load capacity in kg
        """
        # Pressure effect
        pressure_ratio = pressure_kpa / self.max_pressure
        pressure_factor = min(1.0, pressure_ratio)

        # Temperature effect (reduced capacity at high temp)
        temp_factor = 1.0 - 0.002 * max(0, temperature_c - 20.0)
        temp_factor = max(0.8, temp_factor)

        # Structural factor from belt and ply count
        structural_factor = 0.8 + 0.1 * self.belt_count + 0.05 * self.ply_count

        return self.max_load * pressure_factor * temp_factor * structural_factor

    def compute_stiffness(self, temperature_c: float) -> float:
        """Compute tire stiffness.

        Args:
            temperature_c: Temperature in °C

        Returns:
            Stiffness index (0-1)
        """
        # Base stiffness from structure
        base_stiffness = (self.belt_count / 4.0 + self.ply_count / 4.0) / 2.0

        # Temperature effect
        temp_factor = 1.0 - 0.005 * abs(temperature_c - 20.0)
        temp_factor = max(0.5, min(1.2, temp_factor))

        return base_stiffness * temp_factor

    def to_dict(self) -> dict[str, Any]:
        """Serialize structure to dictionary."""
        return {
            "belt_count": self.belt_count,
            "ply_count": self.ply_count,
            "belt_angle": self.belt_angle,
            "carcass_type": self.carcass_type,
            "max_load": self.max_load,
            "max_pressure": self.max_pressure,
        }


@dataclass
class TireGeometry:
    """Complete tire geometry specification.

    Attributes:
        tire_id: Unique identifier
        tire_type: Type of tire
        width: Section width in mm
        aspect_ratio: Aspect ratio (height/width as percentage)
        diameter: Rim diameter in inches
        tread_design: Tread pattern specification
        structure: Internal structure specification
        mass: Total tire mass in kg
    """

    tire_id: str
    tire_type: TireType
    width: float  # mm
    aspect_ratio: float  # percentage
    diameter: float  # inches
    tread_design: TreadDesign
    structure: TireStructure
    mass: float = 10.0  # kg

    def compute_contact_patch_area(self, load_kg: float, pressure_kpa: float) -> float:
        """Compute contact patch area.

        Args:
            load_kg: Applied load in kg
            pressure_kpa: Inflation pressure in kPa

        Returns:
            Contact patch area in cm²
        """
        # Basic contact area from load and pressure
        load_n = load_kg * 9.81
        pressure_pa = pressure_kpa * 1000.0
        area_m2 = load_n / pressure_pa
        area_cm2 = area_m2 * 10000.0

        # Adjustment for tire geometry
        geometry_factor = 1.0 + 0.001 * self.width + 0.01 * self.aspect_ratio

        return area_cm2 * geometry_factor

    def compute_rolling_radius(self, load_kg: float, pressure_kpa: float) -> float:
        """Compute effective rolling radius.

        Args:
            load_kg: Applied load in kg
            pressure_kpa: Inflation pressure in kPa

        Returns:
            Rolling radius in mm
        """
        # Static radius
        sidewall_height = self.width * self.aspect_ratio / 100.0
        rim_radius = self.diameter * 25.4 / 2.0  # Convert inches to mm
        static_radius = rim_radius + sidewall_height

        # Deflection under load
        deflection_factor = 0.98 - 0.0001 * load_kg + 0.0001 * pressure_kpa
        deflection_factor = max(0.90, min(0.99, deflection_factor))

        return static_radius * deflection_factor

    def to_dict(self) -> dict[str, Any]:
        """Serialize geometry to dictionary."""
        return {
            "tire_id": self.tire_id,
            "tire_type": self.tire_type.value,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
            "diameter": self.diameter,
            "mass": self.mass,
            "tread_design": self.tread_design.to_dict(),
            "structure": self.structure.to_dict(),
        }

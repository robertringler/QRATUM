"""Environmental conditions modeling for tire simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RoadSurface(Enum):
    """Road surface types."""

    DRY_ASPHALT = "dry_asphalt"
    WET_ASPHALT = "wet_asphalt"
    DRY_CONCRETE = "dry_concrete"
    WET_CONCRETE = "wet_concrete"
    ICE = "ice"
    SNOW = "snow"
    GRAVEL = "gravel"
    SAND = "sand"
    MUD = "mud"
    TRACK = "track"
    OFF_ROAD = "off_road"
    COBBLESTONE = "cobblestone"


class WeatherCondition(Enum):
    """Weather conditions."""

    CLEAR = "clear"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    ICE = "ice"
    FOG = "fog"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"


@dataclass
class EnvironmentalConditions:
    """Complete environmental conditions for tire simulation.

    Attributes:
        ambient_temperature: Air temperature in °C
        road_temperature: Road surface temperature in °C
        surface_type: Road surface type
        surface_wetness: Surface wetness level (0=dry, 1=fully wet)
        weather: Weather condition
        humidity: Relative humidity (0-1)
        wind_speed: Wind speed in m/s
        altitude: Altitude in meters
        uv_index: UV index (0-11+)
        rainfall_rate: Rainfall rate in mm/h
    """

    ambient_temperature: float = 20.0  # °C
    road_temperature: float = 25.0  # °C
    surface_type: RoadSurface = RoadSurface.DRY_ASPHALT
    surface_wetness: float = 0.0  # 0=dry, 1=wet
    weather: WeatherCondition = WeatherCondition.CLEAR
    humidity: float = 0.5  # 0-1
    wind_speed: float = 0.0  # m/s
    altitude: float = 0.0  # meters
    uv_index: float = 5.0  # UV index
    rainfall_rate: float = 0.0  # mm/h

    def compute_friction_coefficient(self, base_friction: float) -> float:
        """Compute effective friction coefficient for current conditions.

        Args:
            base_friction: Base material friction coefficient

        Returns:
            Effective friction coefficient accounting for environment
        """
        # Surface type effect
        surface_factors = {
            RoadSurface.DRY_ASPHALT: 1.0,
            RoadSurface.WET_ASPHALT: 0.7,
            RoadSurface.DRY_CONCRETE: 0.95,
            RoadSurface.WET_CONCRETE: 0.65,
            RoadSurface.ICE: 0.15,
            RoadSurface.SNOW: 0.3,
            RoadSurface.GRAVEL: 0.6,
            RoadSurface.SAND: 0.5,
            RoadSurface.MUD: 0.4,
            RoadSurface.TRACK: 1.2,
            RoadSurface.OFF_ROAD: 0.7,
            RoadSurface.COBBLESTONE: 0.8,
        }
        surface_factor = surface_factors.get(self.surface_type, 1.0)

        # Wetness effect
        wetness_factor = 1.0 - 0.4 * self.surface_wetness

        # Temperature effect
        temp_factor = 1.0 + 0.005 * (self.road_temperature - 20.0)
        temp_factor = max(0.7, min(1.2, temp_factor))

        return base_friction * surface_factor * wetness_factor * temp_factor

    def compute_thermal_boundary_conditions(self) -> dict[str, float]:
        """Compute thermal boundary conditions for heat transfer simulation.

        Returns:
            Dictionary with convection coefficient and ambient temperature
        """
        # Convection coefficient increases with wind speed
        base_h = 10.0  # W/(m²·K) base convection coefficient
        h_conv = base_h + 5.0 * self.wind_speed

        # Effective ambient temperature
        t_ambient = self.ambient_temperature

        # Solar heating effect
        solar_effect = 0.0
        if self.weather == WeatherCondition.CLEAR or self.weather == WeatherCondition.EXTREME_HEAT:
            solar_effect = 50.0 * (self.uv_index / 10.0)

        return {
            "convection_coefficient": h_conv,
            "ambient_temperature": t_ambient,
            "solar_heating": solar_effect,
        }

    def compute_aging_rate_factor(self) -> float:
        """Compute material aging rate factor for current conditions.

        Returns:
            Aging rate multiplier (1.0 = normal, >1.0 = accelerated)
        """
        # Temperature effect (Arrhenius-like)
        temp_factor = 1.0 + 0.05 * max(0, self.ambient_temperature - 20.0)

        # UV effect
        uv_factor = 1.0 + 0.1 * self.uv_index

        # Humidity effect
        humidity_factor = 1.0 + 0.2 * self.humidity

        # Weather severity
        weather_factors = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.RAIN: 1.1,
            WeatherCondition.HEAVY_RAIN: 1.2,
            WeatherCondition.SNOW: 0.9,
            WeatherCondition.ICE: 0.8,
            WeatherCondition.FOG: 1.05,
            WeatherCondition.EXTREME_HEAT: 1.5,
            WeatherCondition.EXTREME_COLD: 0.7,
        }
        weather_factor = weather_factors.get(self.weather, 1.0)

        return temp_factor * uv_factor * humidity_factor * weather_factor

    def compute_hydroplaning_risk(self, speed_kmh: float, tire_pressure_kpa: float) -> float:
        """Compute hydroplaning risk factor.

        Args:
            speed_kmh: Vehicle speed in km/h
            tire_pressure_kpa: Tire pressure in kPa

        Returns:
            Hydroplaning risk (0=no risk, 1=high risk)
        """
        # Only relevant for wet conditions
        if self.surface_wetness < 0.3:
            return 0.0

        # Speed effect (critical speed around 80-120 km/h depending on conditions)
        speed_factor = max(0, (speed_kmh - 50.0) / 100.0)

        # Wetness and rainfall effect
        water_factor = 0.5 * self.surface_wetness + 0.5 * min(1.0, self.rainfall_rate / 50.0)

        # Pressure effect (higher pressure reduces hydroplaning)
        pressure_factor = max(0.5, 1.0 - (tire_pressure_kpa - 200.0) / 200.0)

        risk = speed_factor * water_factor * pressure_factor
        return min(1.0, risk)

    def to_dict(self) -> dict[str, Any]:
        """Serialize environmental conditions to dictionary."""
        return {
            "ambient_temperature": self.ambient_temperature,
            "road_temperature": self.road_temperature,
            "surface_type": self.surface_type.value,
            "surface_wetness": self.surface_wetness,
            "weather": self.weather.value,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "altitude": self.altitude,
            "uv_index": self.uv_index,
            "rainfall_rate": self.rainfall_rate,
        }

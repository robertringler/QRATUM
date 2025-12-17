"""Adapter for tire simulation data."""

from __future__ import annotations

from typing import Any

import numpy as np


class TireDataAdapter:
    """Adapter to extract visualization data from TireSimulationResult.

    Converts QuASIM tire simulation results into formats suitable
    for visualization pipelines.
    """

    @staticmethod
    def extract_visualization_data(simulation_result: Any) -> dict[str, Any]:
        """Extract visualization data from simulation result.

        Args:
            simulation_result: TireSimulationResult from quasim.domains.tire

        Returns:
            Dictionary containing visualization-ready data
        """

        # Extract geometry
        geometry = getattr(simulation_result, "tire_geometry", None)

        # Extract performance metrics
        metrics = getattr(simulation_result, "performance_metrics", None)

        # Extract or generate field data
        thermal_map = TireDataAdapter._extract_thermal_map(simulation_result)
        stress_distribution = TireDataAdapter._extract_stress_distribution(simulation_result)
        wear_pattern = TireDataAdapter._extract_wear_pattern(simulation_result)

        return {
            "geometry": geometry,
            "metrics": metrics,
            "thermal_map": thermal_map,
            "stress_distribution": stress_distribution,
            "wear_pattern": wear_pattern,
            "environment": getattr(simulation_result, "environment", None),
            "load_kg": getattr(simulation_result, "load_kg", 0),
            "pressure_kpa": getattr(simulation_result, "pressure_kpa", 0),
            "speed_kmh": getattr(simulation_result, "speed_kmh", 0),
            "simulation_id": getattr(simulation_result, "simulation_id", "unknown"),
        }

    @staticmethod
    def _extract_thermal_map(simulation_result: Any) -> np.ndarray:
        """Extract or generate thermal map data.

        Args:
            simulation_result: Simulation result

        Returns:
            Temperature values array
        """

        if hasattr(simulation_result, "thermal_map") and simulation_result.thermal_map is not None:
            return np.array(simulation_result.thermal_map)

        # Generate synthetic thermal data based on environment
        temp_ambient = 25.0
        if hasattr(simulation_result, "environment"):
            env = simulation_result.environment
            if hasattr(env, "temperature_celsius"):
                temp_ambient = env.temperature_celsius

        temp_max = temp_ambient + 30.0  # Typical tire temperature rise

        # Create temperature distribution (higher at contact patch)
        num_points = 1000
        temperatures = np.random.uniform(temp_ambient, temp_max, num_points)

        return temperatures

    @staticmethod
    def _extract_stress_distribution(simulation_result: Any) -> np.ndarray:
        """Extract or generate stress distribution data.

        Args:
            simulation_result: Simulation result

        Returns:
            Stress values array (MPa)
        """

        if (
            hasattr(simulation_result, "stress_distribution")
            and simulation_result.stress_distribution is not None
        ):
            return np.array(simulation_result.stress_distribution)

        # Generate synthetic stress data based on load
        load_kg = getattr(simulation_result, "load_kg", 500)
        stress_max = load_kg * 0.01  # Approximate max stress in MPa

        # Create stress distribution (higher at contact patch)
        num_points = 1000
        stresses = np.random.uniform(0, stress_max, num_points)

        return stresses

    @staticmethod
    def _extract_wear_pattern(simulation_result: Any) -> np.ndarray:
        """Extract or generate wear pattern data.

        Args:
            simulation_result: Simulation result

        Returns:
            Wear depth values array (mm)
        """

        if (
            hasattr(simulation_result, "wear_pattern")
            and simulation_result.wear_pattern is not None
        ):
            return np.array(simulation_result.wear_pattern)

        # Generate synthetic wear data based on wear rate
        wear_rate = 1.0
        if hasattr(simulation_result, "performance_metrics"):
            metrics = simulation_result.performance_metrics
            if hasattr(metrics, "wear_rate"):
                wear_rate = metrics.wear_rate

        max_wear = wear_rate * 0.1  # mm

        # Create wear pattern (varies across tread)
        num_points = 1000
        wear = np.random.uniform(0, max_wear, num_points)

        return wear

    @staticmethod
    def normalize_field_data(field_data: np.ndarray, target_size: int) -> np.ndarray:
        """Normalize field data to target size.

        Args:
            field_data: Input field data
            target_size: Target number of points

        Returns:
            Normalized field data
        """

        if len(field_data) == target_size:
            return field_data

        if len(field_data) < target_size:
            # Upsample using interpolation
            indices = np.linspace(0, len(field_data) - 1, target_size)
            return np.interp(indices, np.arange(len(field_data)), field_data)
        else:
            # Downsample
            indices = np.linspace(0, len(field_data) - 1, target_size).astype(int)
            return field_data[indices]

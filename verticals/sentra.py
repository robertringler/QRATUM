"""SENTRA - Aerospace & Defense Module for QRATUM Platform.

Provides ballistic trajectory simulation, radar cross-section analysis,
orbit propagation, aerodynamic analysis, and mission planning.
"""

import math
from typing import Any, Dict

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  VerticalModuleBase)
from qratum_platform.substrates import VerticalModule, get_optimal_substrate


class SENTRAModule(VerticalModuleBase):
    """Aerospace and Defense module for trajectory and orbital mechanics."""

    MODULE_NAME = "SENTRA"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    SENTRA Aerospace & Defense Disclaimer:
    - FOR RESEARCH AND SIMULATION ONLY
    - NOT for operational military or defense systems
    - Export control regulations may apply (ITAR/EAR)
    - Requires proper authorization for defense applications
    - Models are simplified - not for critical systems
    - Consult qualified aerospace engineers
    """

    PROHIBITED_USES = [
        "weapon targeting without authorization",
        "unauthorized military operations",
        "terrorist applications",
        "export control violations",
        "offensive cyber operations",
    ]

    # Constants
    EARTH_RADIUS_KM = 6371.0
    EARTH_MU = 398600.4418  # km^3/s^2
    G = 9.81  # m/s^2

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute aerospace/defense operation."""
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("sentra_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "trajectory_simulation":
                result = self._simulate_trajectory(parameters)
            elif operation == "radar_analysis":
                result = self._analyze_radar_cross_section(parameters)
            elif operation == "orbit_propagation":
                result = self._propagate_orbit(parameters)
            elif operation == "aerodynamic_analysis":
                result = self._analyze_aerodynamics(parameters)
            elif operation == "mission_planning":
                result = self._plan_mission(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "sentra_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )
            return result
        except Exception as e:
            self.emit_event(
                "sentra_execution_failed",
                contract.contract_id,
                {"operation": operation, "error": str(e)},
            )
            raise

    def _simulate_trajectory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate ballistic trajectory.

        Args:
            parameters: initial_velocity, launch_angle, altitude

        Returns:
            Trajectory simulation results
        """
        v0 = parameters.get("initial_velocity_ms", 500.0)
        angle = parameters.get("launch_angle_deg", 45.0)
        h0 = parameters.get("initial_altitude_m", 0.0)

        angle_rad = math.radians(angle)
        v0x = v0 * math.cos(angle_rad)
        v0y = v0 * math.sin(angle_rad)

        # Time of flight (simplified, no drag)
        t_flight = (v0y + math.sqrt(v0y**2 + 2 * self.G * h0)) / self.G

        # Range
        range_m = v0x * t_flight

        # Max height
        max_height = h0 + (v0y**2) / (2 * self.G)

        # Generate trajectory points
        trajectory = []
        dt = t_flight / 100
        for i in range(101):
            t = i * dt
            x = v0x * t
            y = h0 + v0y * t - 0.5 * self.G * t**2
            if y < 0:
                break
            trajectory.append({"time_s": t, "x_m": x, "y_m": y})

        return {
            "simulation_type": "ballistic_trajectory",
            "time_of_flight_s": t_flight,
            "range_m": range_m,
            "max_height_m": max_height,
            "impact_velocity_ms": math.sqrt(v0x**2 + (v0y - self.G * t_flight) ** 2),
            "trajectory_points": trajectory[::10],  # Downsample
            "note": "Simplified model without drag",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _analyze_radar_cross_section(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze radar cross-section (RCS).

        Args:
            parameters: object_type, frequency_ghz, aspect_angle

        Returns:
            RCS analysis results
        """
        object_type = parameters.get("object_type", "sphere")
        frequency_ghz = parameters.get("frequency_ghz", 10.0)

        wavelength_m = 0.3 / frequency_ghz  # c/f

        # Simplified RCS calculation
        if object_type == "sphere":
            radius_m = parameters.get("radius_m", 1.0)
            rcs_m2 = math.pi * radius_m**2  # Optical region
        elif object_type == "flat_plate":
            area_m2 = parameters.get("area_m2", 1.0)
            rcs_m2 = (4 * math.pi * area_m2**2) / wavelength_m**2
        else:
            rcs_m2 = 1.0  # Default

        # RCS pattern (simplified)
        pattern = []
        for angle in range(0, 360, 30):
            angle_rad = math.radians(angle)
            rcs_at_angle = rcs_m2 * abs(math.cos(angle_rad))
            pattern.append({"angle_deg": angle, "rcs_dbsm": 10 * math.log10(rcs_at_angle + 0.001)})

        return {
            "analysis_type": "radar_cross_section",
            "object_type": object_type,
            "frequency_ghz": frequency_ghz,
            "rcs_m2": rcs_m2,
            "rcs_dbsm": 10 * math.log10(rcs_m2),
            "rcs_pattern": pattern,
            "note": "Simplified RCS model",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _propagate_orbit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate orbital trajectory using two-body mechanics.

        Args:
            parameters: altitude_km, inclination_deg, duration_hours

        Returns:
            Orbit propagation results
        """
        altitude_km = parameters.get("altitude_km", 400.0)
        inclination_deg = parameters.get("inclination_deg", 51.6)
        duration_hours = parameters.get("duration_hours", 24.0)

        # Semi-major axis
        a = self.EARTH_RADIUS_KM + altitude_km

        # Orbital period (Kepler's third law)
        T = 2 * math.pi * math.sqrt(a**3 / self.EARTH_MU)  # seconds

        # Orbital velocity
        v = math.sqrt(self.EARTH_MU / a)

        # Ground track
        num_points = int(duration_hours * 3600 / T) + 1
        ground_track = []
        for i in range(min(num_points, 100)):
            t = i * T
            mean_anomaly = (2 * math.pi * t) / T
            # Simplified ground track
            lon = (mean_anomaly * 180 / math.pi) % 360 - 180
            lat = inclination_deg * math.sin(mean_anomaly)
            ground_track.append({"time_s": t, "latitude_deg": lat, "longitude_deg": lon})

        return {
            "propagation_type": "two_body_keplerian",
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "orbital_period_minutes": T / 60,
            "orbital_velocity_km_s": v,
            "orbits_per_day": 86400 / T,
            "ground_track": ground_track[:24],  # First day samples
            "note": "Simplified two-body mechanics",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _analyze_aerodynamics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze aerodynamic properties.

        Args:
            parameters: shape, velocity_ms, altitude_m

        Returns:
            Aerodynamic analysis results
        """
        shape = parameters.get("shape", "cylinder")
        velocity_ms = parameters.get("velocity_ms", 200.0)
        altitude_m = parameters.get("altitude_m", 10000.0)
        area_m2 = parameters.get("reference_area_m2", 1.0)

        # Atmospheric density (simplified)
        rho = 1.225 * math.exp(-altitude_m / 8500)  # kg/m^3

        # Drag coefficient (simplified)
        cd = {"sphere": 0.47, "cylinder": 1.2, "streamlined": 0.04}.get(shape, 0.5)

        # Dynamic pressure
        q = 0.5 * rho * velocity_ms**2

        # Drag force
        drag_n = cd * area_m2 * q

        # Mach number
        speed_of_sound = 340.0 * math.sqrt(1 - altitude_m / 44300)  # Simplified
        mach = velocity_ms / speed_of_sound

        return {
            "analysis_type": "aerodynamics",
            "shape": shape,
            "mach_number": mach,
            "dynamic_pressure_pa": q,
            "drag_coefficient": cd,
            "drag_force_n": drag_n,
            "lift_to_drag_ratio": 0.0,  # Simplified
            "note": "Simplified CFD - use ANSYS Fluent for production",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _plan_mission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Plan aerospace mission.

        Args:
            parameters: mission_type, waypoints, constraints

        Returns:
            Mission plan
        """
        mission_type = parameters.get("mission_type", "reconnaissance")
        waypoints = parameters.get("waypoints", [{"lat": 0, "lon": 0}, {"lat": 10, "lon": 10}])

        # Calculate total distance
        total_distance_km = 0
        for i in range(len(waypoints) - 1):
            dist = self._calculate_great_circle_distance(waypoints[i], waypoints[i + 1])
            total_distance_km += dist

        # Estimate mission parameters
        avg_speed_kmh = 800.0  # Typical aircraft speed
        mission_duration_hours = total_distance_km / avg_speed_kmh
        fuel_required_kg = total_distance_km * 5.0  # Simplified

        return {
            "mission_type": mission_type,
            "num_waypoints": len(waypoints),
            "total_distance_km": total_distance_km,
            "estimated_duration_hours": mission_duration_hours,
            "estimated_fuel_kg": fuel_required_kg,
            "waypoints": waypoints,
            "note": "Simplified mission planning",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_great_circle_distance(self, p1: Dict, p2: Dict) -> float:
        """Calculate great circle distance between two points."""
        lat1, lon1 = math.radians(p1["lat"]), math.radians(p1["lon"])
        lat2, lon2 = math.radians(p2["lat"]), math.radians(p2["lon"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return self.EARTH_RADIUS_KM * c

    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Get optimal compute substrate for aerospace operation."""
        return get_optimal_substrate(VerticalModule.SENTRA, operation)

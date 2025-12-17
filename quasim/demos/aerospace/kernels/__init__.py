"""Ascent simulation kernel.

Simulates launch vehicle trajectory and dynamics.
"""

from typing import Any, Dict, Optional

import numpy as np

from quasim.common.metrics import rmse
from quasim.viz.run_capture import RunCapture, create_dummy_frame


def simulate_ascent(
    scenario: Dict[str, Any],
    steps: int = 200,
    seed: int = 42,
    capture: Optional[RunCapture] = None,
) -> Dict[str, Any]:
    """Simulate launch vehicle ascent trajectory.

    Args:
        scenario: Scenario configuration
        steps: Number of simulation steps
        seed: Random seed
        capture: Optional run capture for video

    Returns:
        Dictionary with simulation results
    """

    rng = np.random.RandomState(seed)

    # Extract scenario parameters
    mass = scenario.get("mass_kg", 549000)
    thrust = scenario.get("thrust_n", 7607000)
    isp = scenario.get("isp_s", 282)

    # Initialize state
    altitude = 0.0
    velocity = 0.0
    time = 0.0
    dt = 0.5  # seconds

    # Target trajectory
    target_altitude = np.linspace(0, 100000, steps)
    target_velocity = np.linspace(0, 2500, steps)

    # Storage
    altitudes = []
    velocities = []
    q_values = []
    trace = []

    for step in range(steps):
        # Compute acceleration
        g = 9.81 * (1.0 - altitude / 6371000)  # Gravity with altitude correction

        # Atmospheric density (exponential model)
        rho = 1.225 * np.exp(-altitude / 8500)

        # Drag
        cd = 0.3
        area = 50.0  # m^2
        drag = 0.5 * rho * velocity**2 * cd * area

        # Thrust (with some random variation)
        thrust_actual = thrust * (1.0 + 0.01 * rng.randn())

        # Net acceleration
        acc = thrust_actual / mass - g - drag / mass

        # Update state
        velocity += acc * dt
        altitude += velocity * dt
        time += dt

        # Dynamic pressure
        q = 0.5 * rho * velocity**2

        # Store
        altitudes.append(altitude)
        velocities.append(velocity)
        q_values.append(q)

        trace.append(
            {
                "step": step,
                "time": time,
                "altitude": altitude,
                "velocity": velocity,
                "q": q,
                "acceleration": acc,
            }
        )

        # Capture frame if requested
        if capture:
            frame = create_dummy_frame(width=640, height=480, step=step, pattern="gradient")
            capture.record(
                {
                    "frame": frame,
                    "step": step,
                    "altitude": altitude,
                    "velocity": velocity,
                }
            )

    # Compute metrics
    altitudes = np.array(altitudes)
    velocities = np.array(velocities)
    q_values = np.array(q_values)

    rmse_alt = rmse(target_altitude, altitudes)
    rmse_vel = rmse(target_velocity, velocities)
    q_max = np.max(q_values)

    # Fuel margin (simplified)
    fuel_used = steps * dt * thrust / (isp * 9.81)
    fuel_margin = (1.0 - fuel_used / (mass * 0.9)) * 100

    return {
        "rmse_altitude": rmse_alt,
        "rmse_velocity": rmse_vel,
        "q_max": q_max,
        "fuel_margin": max(0, fuel_margin),
        "final_altitude": altitude,
        "final_velocity": velocity,
        "trace": trace,
        "altitudes": altitudes,
        "velocities": velocities,
    }

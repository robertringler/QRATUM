"""Deterministic aerospace dynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AerospaceState:
    altitude: float
    velocity: float
    acceleration: float
    mass: float = 1000.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "altitude": self.altitude,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "mass": self.mass,
        }


class AerospaceDynamics:
    """Simple deterministic kinematics with constant acceleration per step."""

    def __init__(self, timestep: float = 0.1, drag: float = 0.0):
        self._timestep = timestep
        self._drag = drag

    def step(self, state: Dict[str, float]) -> Dict[str, float]:
        altitude = float(state.get("altitude", 0.0))
        velocity = float(state.get("velocity", 0.0))
        acceleration = float(state.get("acceleration", 0.0))
        drag_term = self._drag * velocity
        new_velocity = velocity + (acceleration - drag_term) * self._timestep
        new_altitude = altitude + new_velocity * self._timestep
        return {"altitude": new_altitude, "velocity": new_velocity, "acceleration": acceleration}

    def shock(self, state: Dict[str, float], delta_altitude: float = -5.0) -> Dict[str, float]:
        impacted = dict(state)
        impacted["altitude"] = state.get("altitude", 0.0) + delta_altitude
        impacted["velocity"] = state.get("velocity", 0.0) - delta_altitude
        return impacted

    def rollout(self, state: Dict[str, float], steps: int = 5) -> List[Dict[str, float]]:
        trace: List[Dict[str, float]] = [dict(state)]
        current = dict(state)
        for _ in range(steps):
            current = self.step(current)
            trace.append(current)
        return trace


def aerospace_step(state: Dict[str, float]) -> Dict[str, float]:
    return AerospaceDynamics().step(state)

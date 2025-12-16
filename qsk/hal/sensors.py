"""Sensor abstraction for deterministic feeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Sensor:
    name: str
    generator: Callable[[int], float]

    def read(self, tick: int) -> float:
        return float(self.generator(tick))


def sensor_bundle(sensors: List[Sensor], tick: int) -> Dict[str, float]:
    return {sensor.name: sensor.read(tick) for sensor in sensors}

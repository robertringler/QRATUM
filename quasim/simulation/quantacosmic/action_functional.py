"""Action functional helpers for Quantacosmic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class ActionFunctional:
    """Evaluate a discrete action integral over a temporal contour."""

    lagrangian: Callable[[float, float], float]
    contour: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.contour) < 2:
            raise ValueError("An action functional requires at least two contour points.")

    def evaluate(self, field_samples: Sequence[float]) -> float:
        if len(field_samples) != len(self.contour):
            raise ValueError("Field samples must match contour length.")

        action = 0.0
        for left_index in range(len(self.contour) - 1):
            right_index = left_index + 1
            delta_t = self.contour[right_index] - self.contour[left_index]
            midpoint_field = (field_samples[left_index] + field_samples[right_index]) / 2.0
            midpoint_time = (self.contour[left_index] + self.contour[right_index]) / 2.0
            action += self.lagrangian(midpoint_time, midpoint_field) * delta_t
        return action


__all__ = ["ActionFunctional"]

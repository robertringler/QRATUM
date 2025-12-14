"""Ground simulated state with external observations."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObservationAnchor:
    source: str
    tick: int
    weight: float


@dataclass
class ConfidenceWeights:
    simulation: float = 0.5
    observation: float = 0.5

    def normalize(self) -> "ConfidenceWeights":
        total = self.simulation + self.observation
        if total == 0:
            return ConfidenceWeights(0.5, 0.5)
        return ConfidenceWeights(self.simulation / total, self.observation / total)


@dataclass
class GroundedState:
    """World model state reconciled against observations."""

    blended: dict[str, float] = field(default_factory=dict)
    anchors: dict[str, ObservationAnchor] = field(default_factory=dict)

    @staticmethod
    def reconcile(simulated: dict[str, float], observed: dict[str, float], anchor: ObservationAnchor, weights: ConfidenceWeights) -> "GroundedState":
        weights = weights.normalize()
        blended: dict[str, float] = {}
        for key in sorted(set(simulated.keys()) | set(observed.keys())):
            sim_val = simulated.get(key, 0.0)
            obs_val = observed.get(key, 0.0)
            blended[key] = sim_val * weights.simulation + obs_val * weights.observation
        anchors = {key: anchor for key in observed.keys()}
        return GroundedState(blended=blended, anchors=anchors)

    def divergence(self, simulated: dict[str, float]) -> dict[str, float]:
        return {key: self.blended.get(key, 0.0) - simulated.get(key, 0.0) for key in sorted(self.blended.keys())}

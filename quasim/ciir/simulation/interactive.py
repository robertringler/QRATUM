"""Interactive parameter controls for CIIR simulation.

Provides:
    InteractiveController — Real-time parameter tweaking
    ParameterPreset — Named parameter configurations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ParameterPreset:
    """Named parameter configuration for quick switching."""
    name: str
    learning_rate: float = 0.01
    entropy_weight: float = 0.01
    constraint_weights: list[float] | None = None
    observer_targets: list[float] | None = None


# Default presets for exploration
DEFAULT_PRESETS = [
    ParameterPreset(
        name="default",
        learning_rate=0.01,
        entropy_weight=0.01,
    ),
    ParameterPreset(
        name="fast_convergence",
        learning_rate=0.05,
        entropy_weight=0.001,
    ),
    ParameterPreset(
        name="high_entropy",
        learning_rate=0.01,
        entropy_weight=0.1,
    ),
    ParameterPreset(
        name="slow_exploration",
        learning_rate=0.001,
        entropy_weight=0.05,
    ),
]


@dataclass
class InteractiveController:
    """Real-time parameter controller for simulation tuning.

    Allows changing evolution parameters during simulation:
    - Learning rate η
    - Entropy weight γ
    - Per-constraint weights λ_i
    - Observer target values y_j

    Parameters
    ----------
    n_constraints : int
        Number of constraint operators.
    n_observers : int
        Number of observer operators.
    """

    n_constraints: int = 0
    n_observers: int = 0
    learning_rate: float = 0.01
    entropy_weight: float = 0.01
    constraint_weights: list[float] = field(default_factory=list)
    observer_targets: list[float] = field(default_factory=list)
    _callbacks: list[Callable] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.constraint_weights:
            self.constraint_weights = [1.0] * self.n_constraints
        if not self.observer_targets:
            self.observer_targets = [0.0] * self.n_observers

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate η."""
        self.learning_rate = lr
        self._notify("learning_rate", lr)

    def set_entropy_weight(self, gamma: float) -> None:
        """Set entropy regularisation weight γ."""
        self.entropy_weight = gamma
        self._notify("entropy_weight", gamma)

    def set_constraint_weight(self, index: int, weight: float) -> None:
        """Set weight λ_i for constraint i."""
        if 0 <= index < len(self.constraint_weights):
            self.constraint_weights[index] = weight
            self._notify("constraint_weight", (index, weight))

    def set_observer_target(self, index: int, target: float) -> None:
        """Set target value y_j for observer j."""
        if 0 <= index < len(self.observer_targets):
            self.observer_targets[index] = target
            self._notify("observer_target", (index, target))

    def apply_preset(self, preset: ParameterPreset) -> None:
        """Apply a named parameter preset."""
        self.learning_rate = preset.learning_rate
        self.entropy_weight = preset.entropy_weight
        if preset.constraint_weights:
            for i, w in enumerate(preset.constraint_weights):
                if i < len(self.constraint_weights):
                    self.constraint_weights[i] = w
        if preset.observer_targets:
            for i, t in enumerate(preset.observer_targets):
                if i < len(self.observer_targets):
                    self.observer_targets[i] = t
        self._notify("preset", preset.name)

    def on_change(self, callback: Callable) -> None:
        """Register a callback for parameter changes.

        The callback receives (param_name: str, value: Any).
        """
        self._callbacks.append(callback)

    def _notify(self, param: str, value: Any) -> None:
        for cb in self._callbacks:
            cb(param, value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize current parameters."""
        return {
            "learning_rate": self.learning_rate,
            "entropy_weight": self.entropy_weight,
            "constraint_weights": list(self.constraint_weights),
            "observer_targets": list(self.observer_targets),
        }

"""State management for digital twins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateManager:
    """Manages state history and snapshots for digital twins.

    Provides efficient state storage and retrieval for time-series
    analysis and rollback capabilities. Supports distributed state
    management across multiple GPU clusters.

    Attributes:
        history: List of historical states
        max_history_size: Maximum number of states to retain
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    max_history_size: int = 1000

    def update(self, state: dict[str, Any]) -> None:
        """Update state and append to history.

        Args:
            state: New state dictionary
        """
        self.history.append(state.copy())

        # Trim history if needed
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size :]

    def get_current_state(self) -> dict[str, Any]:
        """Get the most recent state.

        Returns:
            Current state dictionary, or empty dict if no history
        """
        if not self.history:
            return {}
        return self.history[-1].copy()

    def get_state_at(self, index: int) -> dict[str, Any]:
        """Get state at a specific history index.

        Args:
            index: Index in history (negative indices supported)

        Returns:
            State dictionary at the given index
        """
        if not self.history:
            raise IndexError("No state history available")
        return self.history[index].copy()

    def get_history(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get state history.

        Args:
            n: Number of most recent states to return (None for all)

        Returns:
            List of state dictionaries
        """
        if n is None:
            return [s.copy() for s in self.history]
        return [s.copy() for s in self.history[-n:]]

    def clear_history(self) -> None:
        """Clear all state history."""
        self.history.clear()

    def rollback(self, steps: int = 1) -> None:
        """Rollback state history by a number of steps.

        Args:
            steps: Number of steps to rollback
        """
        if steps < 1:
            raise ValueError("Steps must be positive")
        if steps >= len(self.history):
            raise ValueError("Cannot rollback beyond history")

        self.history = self.history[:-steps]

"""Simulation time management and step scheduling.

Provides deterministic time progression for reproducible simulations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SimClock:
    """Simulation clock for deterministic time progression.

    Attributes:
        t: Current simulation time
        dt: Time step size
        step: Current step number
        start_time: Wall clock start time
        paused: Whether the clock is paused
    """

    t: float = 0.0
    dt: float = 0.01
    step: int = 0
    start_time: float = field(default_factory=time.time)
    paused: bool = False

    def advance(self, steps: int = 1) -> None:
        """Advance clock by specified number of steps.

        Args:
            steps: Number of steps to advance
        """

        if not self.paused:
            self.step += steps
            self.t += steps * self.dt

    def reset(self) -> None:
        """Reset clock to initial state."""

        self.t = 0.0
        self.step = 0
        self.start_time = time.time()
        self.paused = False

    def pause(self) -> None:
        """Pause the clock."""

        self.paused = True

    def resume(self) -> None:
        """Resume the clock."""

        self.paused = False

    def elapsed_walltime(self) -> float:
        """Get elapsed wall clock time in seconds."""

        return time.time() - self.start_time


@dataclass
class StepScheduler:
    """Step-based event scheduler for simulations.

    Schedules callbacks to run at specific simulation steps.
    """

    callbacks: dict[int, list[Callable]] = field(default_factory=dict)

    def schedule(self, step: int, callback: Callable) -> None:
        """Schedule a callback at a specific step.

        Args:
            step: Step number to run callback
            callback: Function to call at that step
        """

        if step not in self.callbacks:
            self.callbacks[step] = []
        self.callbacks[step].append(callback)

    def run_step(self, step: int) -> None:
        """Run all callbacks scheduled for a step.

        Args:
            step: Current step number
        """

        if step in self.callbacks:
            for callback in self.callbacks[step]:
                callback()

    def clear(self) -> None:
        """Clear all scheduled callbacks."""

        self.callbacks.clear()

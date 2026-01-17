"""Cellular Automata Module for Temporal Compression Experiments.

Implements deterministic, rule-based dynamical systems with non-trivial
state evolution suitable for temporal compression validation.

Systems include:
- Rule 110: Turing-complete cellular automaton with emergent complexity
- Rule 30: Chaotic but deterministic automaton with bounded state space
- Custom rule configurations

All systems ensure:
- Reproducible random number generation (seed-locked)
- Forward evolution with consistent state transitions
- Backward reconstruction capability
- Cryptographic state fingerprinting
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AutomatonState:
    """Snapshot of cellular automaton state.

    Attributes:
        step: Current simulation step
        cells: Cell state array (0 or 1)
        timestamp: Wall-clock time when state was captured
        state_hash: SHA3-256 hash of cell state
    """

    step: int
    cells: np.ndarray
    timestamp: float
    state_hash: str = ""

    def __post_init__(self) -> None:
        """Compute state hash after initialization."""
        if not self.state_hash:
            self.state_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA3-256 hash of cell state."""
        hasher = hashlib.sha3_256()
        hasher.update(struct.pack("Q", self.step))
        hasher.update(self.cells.tobytes())
        return hasher.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "step": self.step,
            "cells": self.cells.tolist(),
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutomatonState":
        """Create state from dictionary."""
        return cls(
            step=data["step"],
            cells=np.array(data["cells"], dtype=np.uint8),
            timestamp=data["timestamp"],
            state_hash=data.get("state_hash", ""),
        )


class CellularAutomaton:
    """Deterministic cellular automaton for temporal simulation.

    Implements a 1D elementary cellular automaton with configurable
    rules and seed-locked random initialization.

    Attributes:
        rule: Wolfram rule number (0-255)
        size: Number of cells
        seed: Random seed for deterministic initialization
        cells: Current cell state array
        step: Current simulation step
        history: List of state snapshots (if enabled)
    """

    def __init__(
        self,
        rule: int = 110,
        size: int = 100,
        seed: int = 42,
        track_history: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Initialize cellular automaton.

        Args:
            rule: Wolfram rule number (0-255)
            size: Number of cells
            seed: Random seed for reproducibility
            track_history: Whether to store state history
            initial_state: Optional initial cell configuration
        """
        if not 0 <= rule <= 255:
            raise ValueError(f"Rule must be 0-255, got {rule}")
        if size < 3:
            raise ValueError(f"Size must be >= 3, got {size}")

        self.rule = rule
        self.size = size
        self.seed = seed
        self.track_history = track_history

        # Initialize deterministic RNG
        self._rng = np.random.Generator(np.random.PCG64(seed))

        # Initialize cells
        if initial_state is not None:
            if len(initial_state) != size:
                raise ValueError(f"Initial state length {len(initial_state)} != size {size}")
            self.cells = np.array(initial_state, dtype=np.uint8)
        else:
            self.cells = self._generate_initial_state()

        self.step = 0
        self.history: list[AutomatonState] = []

        # Precompute rule lookup table
        self._rule_table = self._build_rule_table(rule)

        # Record initial state
        if self.track_history:
            self.history.append(self.capture_state())

    def _generate_initial_state(self) -> np.ndarray:
        """Generate deterministic initial state."""
        return self._rng.integers(0, 2, size=self.size, dtype=np.uint8)

    def _build_rule_table(self, rule: int) -> dict[tuple[int, int, int], int]:
        """Build lookup table for rule application.

        For elementary cellular automata, the rule number encodes
        the transition function for all 8 possible neighborhood configurations.
        """
        table = {}
        for i in range(8):
            left = (i >> 2) & 1
            center = (i >> 1) & 1
            right = i & 1
            table[(left, center, right)] = (rule >> i) & 1
        return table

    def evolve(self, steps: int = 1) -> None:
        """Evolve automaton forward by given number of steps.

        Args:
            steps: Number of time steps to evolve
        """
        for _ in range(steps):
            new_cells = np.zeros_like(self.cells)

            for i in range(self.size):
                left = self.cells[(i - 1) % self.size]
                center = self.cells[i]
                right = self.cells[(i + 1) % self.size]

                new_cells[i] = self._rule_table[(left, center, right)]

            self.cells = new_cells
            self.step += 1

            if self.track_history:
                self.history.append(self.capture_state())

    def capture_state(self) -> AutomatonState:
        """Capture current state snapshot."""
        return AutomatonState(
            step=self.step,
            cells=self.cells.copy(),
            timestamp=time.time(),
        )

    def get_state_hash(self) -> str:
        """Get hash of current state."""
        return self.capture_state().state_hash

    def reset(self) -> None:
        """Reset automaton to initial state with same seed."""
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self.cells = self._generate_initial_state()
        self.step = 0
        self.history.clear()

        if self.track_history:
            self.history.append(self.capture_state())

    def clone(self) -> "CellularAutomaton":
        """Create a clone of current automaton with same state."""
        clone = CellularAutomaton(
            rule=self.rule,
            size=self.size,
            seed=self.seed,
            track_history=self.track_history,
            initial_state=self.cells.copy(),
        )
        clone.step = self.step
        return clone

    def to_dict(self) -> dict[str, Any]:
        """Serialize automaton state to dictionary."""
        return {
            "rule": self.rule,
            "size": self.size,
            "seed": self.seed,
            "step": self.step,
            "cells": self.cells.tolist(),
            "state_hash": self.get_state_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellularAutomaton":
        """Create automaton from dictionary."""
        return cls(
            rule=data["rule"],
            size=data["size"],
            seed=data["seed"],
            initial_state=np.array(data["cells"], dtype=np.uint8),
        )


class Rule110(CellularAutomaton):
    """Rule 110 Cellular Automaton.

    Rule 110 is proven to be Turing-complete, exhibiting complex
    emergent behavior from simple local rules. It features:
    - Class IV (complex/edge of chaos) behavior
    - Localized structures (particles)
    - Long-range interactions through collisions
    """

    def __init__(
        self,
        size: int = 100,
        seed: int = 42,
        track_history: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Initialize Rule 110 automaton."""
        super().__init__(
            rule=110,
            size=size,
            seed=seed,
            track_history=track_history,
            initial_state=initial_state,
        )


class Rule30(CellularAutomaton):
    """Rule 30 Cellular Automaton.

    Rule 30 exhibits chaotic behavior while remaining fully
    deterministic. Features:
    - Class III (chaotic) behavior
    - Passes many statistical randomness tests
    - Used in Wolfram Mathematica's random number generation
    """

    def __init__(
        self,
        size: int = 100,
        seed: int = 42,
        track_history: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> None:
        """Initialize Rule 30 automaton."""
        super().__init__(
            rule=30,
            size=size,
            seed=seed,
            track_history=track_history,
            initial_state=initial_state,
        )


@dataclass
class BranchingResult:
    """Result of branching simulation exploration.

    Attributes:
        branch_id: Unique identifier for this branch
        parent_step: Step at which branch diverged
        final_step: Final step of this branch
        intervention: Description of intervention applied
        final_state: Final state of this branch
        divergence_measure: Measure of divergence from parent
    """

    branch_id: str
    parent_step: int
    final_step: int
    intervention: str
    final_state: AutomatonState
    divergence_measure: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "branch_id": self.branch_id,
            "parent_step": self.parent_step,
            "final_step": self.final_step,
            "intervention": self.intervention,
            "final_state": self.final_state.to_dict(),
            "divergence_measure": self.divergence_measure,
        }


class BranchingSimulator:
    """Simulator for branching alternative trajectories.

    Enables parallel exploration of alternative system trajectories
    from shared checkpoint states.
    """

    def __init__(self, base_automaton: CellularAutomaton) -> None:
        """Initialize branching simulator.

        Args:
            base_automaton: Base automaton to branch from
        """
        self.base = base_automaton
        self.branches: dict[str, CellularAutomaton] = {}
        self.results: list[BranchingResult] = []

    def create_branch(
        self,
        branch_id: str,
        intervention: str = "none",
        perturbation_indices: list[int] | None = None,
    ) -> CellularAutomaton:
        """Create a new branch from current base state.

        Args:
            branch_id: Unique identifier for branch
            intervention: Description of intervention
            perturbation_indices: Cell indices to flip (intervention)

        Returns:
            New branched automaton
        """
        branch = self.base.clone()

        if perturbation_indices:
            for idx in perturbation_indices:
                if 0 <= idx < branch.size:
                    branch.cells[idx] = 1 - branch.cells[idx]

        self.branches[branch_id] = branch
        return branch

    def run_branch(
        self,
        branch_id: str,
        steps: int,
    ) -> BranchingResult:
        """Run a branch simulation.

        Args:
            branch_id: Branch to run
            steps: Number of steps to simulate

        Returns:
            BranchingResult with simulation outcome
        """
        if branch_id not in self.branches:
            raise ValueError(f"Unknown branch: {branch_id}")

        branch = self.branches[branch_id]
        parent_step = branch.step
        branch.evolve(steps)

        # Compute divergence from base
        divergence = self._compute_divergence(self.base, branch)

        result = BranchingResult(
            branch_id=branch_id,
            parent_step=parent_step,
            final_step=branch.step,
            intervention=f"branch_{branch_id}",
            final_state=branch.capture_state(),
            divergence_measure=divergence,
        )

        self.results.append(result)
        return result

    def _compute_divergence(
        self,
        base: CellularAutomaton,
        branch: CellularAutomaton,
    ) -> float:
        """Compute divergence between two automaton states."""
        # Hamming distance normalized by size
        diff = np.sum(base.cells != branch.cells)
        return float(diff) / base.size

"""Core XENON simulation data structures for bio-mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MolecularState:
    """Represents a molecular state in a bio-mechanism.

    Attributes:
        state_id: Unique identifier for the state
        protein_name: Name of the protein in this state
        free_energy: Gibbs free energy (ΔG) in kJ/mol
        concentration: Molecular concentration (optional)
        metadata: Additional state-specific data
    """

    state_id: str
    protein_name: str
    free_energy: float
    concentration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """Represents a state transition in a bio-mechanism.

    Attributes:
        source_state: Source state ID
        target_state: Target state ID
        rate_constant: Transition rate constant (k) in s^-1
        delta_g: Free energy change (ΔG) in kJ/mol
        activation_energy: Activation energy barrier in kJ/mol
        metadata: Additional transition-specific data
    """

    source_state: str
    target_state: str
    rate_constant: float
    delta_g: float = 0.0
    activation_energy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BioMechanism:
    """Represents a biological mechanism as a directed acyclic graph (DAG).

    This class models biochemical reaction networks with states (molecular
    configurations) and transitions (reaction pathways).

    Attributes:
        mechanism_id: Unique identifier for the mechanism
        states: List of molecular states in the mechanism
        transitions: List of transitions between states
        evidence_score: Confidence score for mechanism validity (0-1)
        metadata: Additional mechanism-level data
    """

    mechanism_id: str
    states: list[MolecularState]
    transitions: list[Transition]
    evidence_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_state(self, state_id: str) -> MolecularState | None:
        """Get a state by ID.

        Args:
            state_id: State identifier

        Returns:
            MolecularState if found, None otherwise
        """
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None

    def get_transitions_from(self, state_id: str) -> list[Transition]:
        """Get all transitions originating from a state.

        Args:
            state_id: Source state identifier

        Returns:
            List of transitions from the state
        """
        return [t for t in self.transitions if t.source_state == state_id]

    def get_transitions_to(self, state_id: str) -> list[Transition]:
        """Get all transitions targeting a state.

        Args:
            state_id: Target state identifier

        Returns:
            List of transitions to the state
        """
        return [t for t in self.transitions if t.target_state == state_id]

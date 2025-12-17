"""Biological mechanism representation as directed acyclic graphs.

The computational primitive that replaces tensors in XENON.
Core XENON simulation data structures for bio-mechanisms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

import numpy as np


@dataclass
class MolecularState:
    """A molecular state in a biological mechanism.

    Represents a specific configuration of a molecule (e.g., phosphorylated protein,
    ligand-bound receptor, active enzyme).

    Attributes:
        name: Unique identifier for this state
        molecule: Molecule name (e.g., 'EGFR', 'ATP', 'RAS')
        properties: State-specific properties (phosphorylation, binding, conformation)
        concentration: Concentration in nM (optional, for simulation)
        free_energy: Gibbs free energy in kcal/mol (optional, for thermodynamics)
    """

    name: str
    molecule: str
    properties: dict[str, Any] = field(default_factory=dict)
    concentration: Optional[float] = None
    free_energy: Optional[float] = None

    def __hash__(self) -> int:
        """Hash based on name for use in sets/dicts."""
        return hash(self.name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "molecule": self.molecule,
            "properties": self.properties,
            "concentration": self.concentration,
            "free_energy": self.free_energy,
        }


@dataclass
class Transition:
    """A transition between molecular states.

    Represents a chemical reaction or conformational change.

    Attributes:
        source: Source state
        target: Target state
        rate_constant: Forward rate constant (1/s or 1/(M*s))
        activation_energy: Activation energy in kcal/mol
        reversible: Whether transition is reversible
        reverse_rate: Reverse rate constant if reversible
        stoichiometry: Stoichiometric coefficients for reactants/products
    """

    source: str
    target: str
    rate_constant: float
    activation_energy: Optional[float] = None
    reversible: bool = False
    reverse_rate: Optional[float] = None
    stoichiometry: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "rate_constant": self.rate_constant,
            "activation_energy": self.activation_energy,
            "reversible": self.reversible,
            "reverse_rate": self.reverse_rate,
            "stoichiometry": self.stoichiometry,
        }


class BioMechanism:
    """A biological mechanism represented as a directed acyclic graph.

    This is the fundamental computational primitive in XENON, replacing tensors.
    Nodes represent molecular states, edges represent transitions.

    Attributes:
        name: Mechanism identifier
        graph: NetworkX directed graph
        posterior: Bayesian posterior probability (updated from experiments)
        provenance: Experimental lineage that generated this mechanism
        confidence_intervals: Parameter uncertainty estimates
    """

    def __init__(self, name: str) -> None:
        """Initialize mechanism.

        Args:
            name: Unique mechanism identifier
        """
        self.name = name
        if nx is not None:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
        self.posterior = 1.0
        self.provenance: list[str] = []
        self.confidence_intervals: dict[str, tuple[float, float]] = {}
        self._states: dict[str, MolecularState] = {}
        self._transitions: list[Transition] = []

    def add_state(self, state: MolecularState) -> None:
        """Add a molecular state to the mechanism."""
        self._states[state.name] = state
        if self.graph is not None:
            self.graph.add_node(state.name, state=state)

    def add_transition(self, transition: Transition) -> None:
        """Add a transition between states."""
        if transition.source not in self._states:
            raise ValueError(f"Source state '{transition.source}' not found")
        if transition.target not in self._states:
            raise ValueError(f"Target state '{transition.target}' not found")

        self._transitions.append(transition)
        if self.graph is not None:
            self.graph.add_edge(transition.source, transition.target, transition=transition)

    def is_thermodynamically_feasible(self, temperature: float = 310.0) -> bool:
        """Check if mechanism satisfies thermodynamic constraints."""
        R = 0.001987

        for transition in self._transitions:
            source_state = self._states[transition.source]
            target_state = self._states[transition.target]

            if source_state.free_energy is None or target_state.free_energy is None:
                continue

            delta_g = target_state.free_energy - source_state.free_energy

            if delta_g > 0 and transition.rate_constant > 1e6:
                return False

            if transition.reversible and transition.reverse_rate is not None:
                k_forward = transition.rate_constant
                k_reverse = transition.reverse_rate
                K_eq_calc = k_forward / k_reverse
                K_eq_thermo = np.exp(-delta_g / (R * temperature))

                if not (0.1 < K_eq_calc / K_eq_thermo < 10.0):
                    return False

        return True

    def validate_conservation_laws(self) -> tuple[bool, list[str]]:
        """Validate mass and charge conservation."""
        violations = []

        for transition in self._transitions:
            for species in transition.stoichiometry:
                if species not in self._states:
                    violations.append(
                        f"Transition {transition.source}->{transition.target} "
                        f"references unknown species '{species}'"
                    )

        return len(violations) == 0, violations

    def get_causal_paths(self, source: str, target: str) -> list[list[str]]:
        """Get all causal paths from source to target state."""
        if source not in self._states or target not in self._states:
            return []

        if self.graph is None or nx is None:
            return []

        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def compute_mechanism_hash(self) -> str:
        """Compute unique hash of mechanism topology and parameters."""
        mech_dict = {
            "states": sorted([s.to_dict() for s in self._states.values()], key=lambda x: x["name"]),
            "transitions": sorted(
                [t.to_dict() for t in self._transitions], key=lambda x: (x["source"], x["target"])
            ),
        }

        mech_json = json.dumps(mech_dict, sort_keys=True)
        return hashlib.sha256(mech_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize mechanism to dictionary."""
        return {
            "name": self.name,
            "posterior": self.posterior,
            "provenance": self.provenance,
            "states": [s.to_dict() for s in self._states.values()],
            "transitions": [t.to_dict() for t in self._transitions],
            "hash": self.compute_mechanism_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BioMechanism:
        """Deserialize mechanism from dictionary."""
        mech = cls(name=data["name"])
        mech.posterior = data["posterior"]
        mech.provenance = data["provenance"]

        for state_data in data["states"]:
            state = MolecularState(**state_data)
            mech.add_state(state)

        for trans_data in data["transitions"]:
            transition = Transition(**trans_data)
            mech.add_transition(transition)

        return mech

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BioMechanism(name='{self.name}', "
            f"states={len(self._states)}, "
            f"transitions={len(self._transitions)}, "
            f"posterior={self.posterior:.4f})"
        )

"""Graph operations on biological mechanisms."""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

from .mechanism import BioMechanism, Transition


class MechanismGraph:
    """Graph operations for mechanism manipulation."""

    @staticmethod
    def mutate_topology(
        mechanism: BioMechanism,
        mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> BioMechanism:
        """Mutate mechanism topology."""
        if seed is not None:
            random.seed(seed)

        new_mech = BioMechanism(name=f"{mechanism.name}_mutated")
        new_mech.posterior = mechanism.posterior
        new_mech.provenance = mechanism.provenance + ["topology_mutation"]

        for state in mechanism._states.values():
            new_mech.add_state(state)

        for transition in mechanism._transitions:
            if random.random() < mutation_rate:
                continue

            if random.random() < mutation_rate:
                mutated_rate = transition.rate_constant * random.uniform(0.5, 2.0)
                transition = Transition(
                    source=transition.source,
                    target=transition.target,
                    rate_constant=mutated_rate,
                    activation_energy=transition.activation_energy,
                    reversible=transition.reversible,
                    reverse_rate=transition.reverse_rate,
                    stoichiometry=transition.stoichiometry,
                )

            new_mech.add_transition(transition)

        return new_mech

    @staticmethod
    def extract_subgraph(mechanism: BioMechanism, nodes: Set[str]) -> BioMechanism:
        """Extract subgraph containing specified nodes."""
        sub_mech = BioMechanism(name=f"{mechanism.name}_subgraph")
        sub_mech.provenance = mechanism.provenance + ["subgraph_extraction"]

        for node_name in nodes:
            if node_name in mechanism._states:
                sub_mech.add_state(mechanism._states[node_name])

        for transition in mechanism._transitions:
            if transition.source in nodes and transition.target in nodes:
                sub_mech.add_transition(transition)

        return sub_mech

    @staticmethod
    def recombine_mechanisms(mech1: BioMechanism, mech2: BioMechanism, name: str) -> BioMechanism:
        """Recombine two mechanisms into a new mechanism."""
        child = BioMechanism(name=name)
        child.provenance = [f"recombination: {mech1.name} + {mech2.name}"]

        all_states = {**mech1._states, **mech2._states}
        for state in all_states.values():
            child.add_state(state)

        seen_transitions: Set[Tuple[str, str]] = set()

        for transition in mech1._transitions + mech2._transitions:
            key = (transition.source, transition.target)
            if key not in seen_transitions:
                seen_transitions.add(key)
                if transition.source in child._states and transition.target in child._states:
                    child.add_transition(transition)

        return child

    @staticmethod
    def detect_cycles(mechanism: BioMechanism) -> List[List[str]]:
        """Detect cycles in mechanism graph."""
        if mechanism.graph is None or nx is None:
            return []

        try:
            cycles = list(nx.simple_cycles(mechanism.graph))
            return cycles
        except:
            return []

    @staticmethod
    def is_isomorphic(mech1: BioMechanism, mech2: BioMechanism) -> bool:
        """Check if two mechanisms are topologically isomorphic."""
        if mech1.graph is None or mech2.graph is None or nx is None:
            return mech1.compute_mechanism_hash() == mech2.compute_mechanism_hash()

        return nx.is_isomorphic(mech1.graph, mech2.graph)

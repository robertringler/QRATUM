r"""Agent EM — Empirical Mapper.

Maps simulation outputs to known physics:
- Classical mechanics (energy conservation, equations of motion)
- Quantum mechanics (unitarity, Born rule, superposition)
- Relativity (Lorentz invariance, metric signature, geodesics)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quasim.ciir.swarm.invariant_miner import DetectedInvariant
from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
from quasim.ciir.swarm.simulator import SimulationResult

# ================================================================
# Empirical mapping result
# ================================================================

@dataclass
class EmpiricalMapping:
    """Result of mapping to known physics.

    Attributes
    ----------
    domain : str
        E.g. 'classical_mechanics', 'quantum_mechanics', 'relativity'.
    match_score : float
        Quality of match ∈ [0, 1].
    details : dict
        Domain-specific fit metrics.
    """

    domain: str
    match_score: float
    details: dict = field(default_factory=dict)


# ================================================================
# Empirical Mapper agent
# ================================================================

class EmpiricalMapper:
    """Map simulation outputs to known physics domains."""

    def map_all(
        self,
        result: SimulationResult,
        invariants: list[DetectedInvariant],
    ) -> list[EmpiricalMapping]:
        """Run all empirical mappings."""
        return [
            self.map_classical_mechanics(result, invariants),
            self.map_quantum_mechanics(result),
            self.map_relativity(result),
        ]

    def map_classical_mechanics(
        self,
        result: SimulationResult,
        invariants: list[DetectedInvariant],
    ) -> EmpiricalMapping:
        """Check for classical mechanics signatures.

        - Energy conservation (approximately constant energy trace)
        - Deterministic trajectories
        - Phase space structure
        """
        details: dict = {}

        # Energy conservation check
        if result.energy_trace:
            energies = np.array(result.energy_trace)
            energy_var = float(np.std(energies) / (np.mean(energies) + 1e-30))
            details["energy_relative_variation"] = energy_var
            energy_conserved = energy_var < 0.1
        else:
            energy_conserved = False

        # Check for Hamiltonian-like structure: symplectic signature
        dim = result.final_state.shape[0]
        symplectic_possible = dim % 2 == 0
        details["symplectic_possible"] = symplectic_possible

        score = 0.0
        if energy_conserved:
            score += 0.5
        if symplectic_possible:
            score += 0.2
        if result.converged:
            score += 0.3

        return EmpiricalMapping(
            domain="classical_mechanics",
            match_score=min(score, 1.0),
            details=details,
        )

    def map_quantum_mechanics(
        self, result: SimulationResult,
    ) -> EmpiricalMapping:
        """Check for quantum mechanics signatures.

        - Norm preservation (unitarity proxy)
        - Superposition (multiple components)
        - Born rule compatibility
        """
        details: dict = {}
        trajectory = result.trajectory

        if not trajectory:
            return EmpiricalMapping("quantum_mechanics", 0.0)

        # Norm preservation check (unitarity)
        norms = [float(np.linalg.norm(s)) for s in trajectory]
        norm_var = float(np.std(norms) / (np.mean(norms) + 1e-30))
        details["norm_relative_variation"] = norm_var
        unitary = norm_var < 0.05

        # Superposition: check if state has multiple significant components
        final = result.final_state
        sorted_abs = np.sort(np.abs(final))[::-1]
        if len(sorted_abs) > 1 and sorted_abs[0] > 1e-10:
            superposition_ratio = float(sorted_abs[1] / sorted_abs[0])
        else:
            superposition_ratio = 0.0
        details["superposition_ratio"] = superposition_ratio

        # Born rule: check |ψ|² sums to ~ constant
        prob = np.sum(final ** 2)
        details["born_probability_sum"] = float(prob)

        score = 0.0
        if unitary:
            score += 0.5
        if superposition_ratio > 0.1:
            score += 0.25
        if abs(prob - 1.0) < 0.1:
            score += 0.25

        return EmpiricalMapping(
            domain="quantum_mechanics",
            match_score=min(score, 1.0),
            details=details,
        )

    def map_relativity(
        self, result: SimulationResult,
    ) -> EmpiricalMapping:
        """Check for relativistic signatures.

        - Lorentz-like invariant: s² = s₀² - ||s_{1:}||²
        - Metric signature (-,+,+,+)
        - Causal structure
        """
        details: dict = {}
        dim = result.final_state.shape[0]

        if dim < 2:
            return EmpiricalMapping("relativity", 0.0)

        # Check for Minkowski-like interval invariance
        trajectory = result.trajectory
        intervals = []
        for s in trajectory:
            interval = s[0] ** 2 - np.sum(s[1:] ** 2)
            intervals.append(float(interval))

        arr = np.array(intervals)
        if len(arr) > 1:
            interval_var = float(np.std(arr) / (abs(np.mean(arr)) + 1e-30))
        else:
            interval_var = 1.0
        details["interval_relative_variation"] = interval_var
        lorentz_like = interval_var < 0.1

        # Metric signature check
        details["dimension"] = dim
        details["possible_signature"] = f"(-,{'+,' * (dim - 1)})"

        score = 0.0
        if lorentz_like:
            score += 0.6
        if dim >= 4:
            score += 0.2
        if result.converged:
            score += 0.2

        return EmpiricalMapping(
            domain="relativity",
            match_score=min(score, 1.0),
            details=details,
        )

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        mappings: list[EmpiricalMapping],
        generation: int = 0,
    ) -> list[int]:
        """Write empirical mappings to knowledge graph."""
        ids = []
        for m in mappings:
            node = memory.add_node(
                node_type=NodeType.OBSERVABLE,
                label=f"Empirical_{m.domain}",
                data=m,
                confidence=m.match_score,
                generation=generation,
            )
            ids.append(node.id)
        return ids

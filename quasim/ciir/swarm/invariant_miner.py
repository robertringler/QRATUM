r"""Agent IM — Invariant Miner.

Detects conserved quantities, symmetries, and attractors from
simulation trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.memory import KnowledgeGraph, NodeType
from quasim.ciir.swarm.simulator import SimulationResult

# ================================================================
# Invariant types
# ================================================================

@dataclass
class DetectedInvariant:
    """A conserved quantity extracted from trajectories.

    Attributes
    ----------
    name : str
    compute : Callable[[NDArray], float]
        Extracts scalar value from state.
    values : list of float
        Time series of the quantity.
    relative_variation : float
        std / |mean| — low means well-conserved.
    """

    name: str
    compute: Callable[[NDArray], float]
    values: list[float] = field(default_factory=list)
    relative_variation: float = 0.0


@dataclass
class DetectedAttractor:
    """An attractor (fixed point or cycle) found in trajectories.

    Attributes
    ----------
    attractor_type : str
        'fixed_point', 'limit_cycle', or 'strange'.
    state : NDArray or list of NDArray
        The attractor state(s).
    basin_size : int
        Number of initial conditions converging here.
    """

    attractor_type: str
    state: NDArray | list[NDArray] = field(default_factory=lambda: np.zeros(1))
    basin_size: int = 1


# ================================================================
# Invariant Miner agent
# ================================================================

class InvariantMiner:
    """Detect conserved quantities and symmetries."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    def mine_invariants(
        self, result: SimulationResult,
    ) -> list[DetectedInvariant]:
        """Detect conserved quantities from a simulation trajectory."""
        trajectory = result.trajectory
        if len(trajectory) < 2:
            return []

        invariants = []

        # 1. Total energy (norm squared)
        energy_inv = self._check_invariant(
            "Energy",
            lambda s: float(np.sum(s ** 2)),
            trajectory,
        )
        invariants.append(energy_inv)

        # 2. Norm (L2)
        norm_inv = self._check_invariant(
            "Norm",
            lambda s: float(np.linalg.norm(s)),
            trajectory,
        )
        invariants.append(norm_inv)

        # 3. Centre of mass (mean component)
        com_inv = self._check_invariant(
            "CentreOfMass",
            lambda s: float(np.mean(s)),
            trajectory,
        )
        invariants.append(com_inv)

        # 4. Max component
        max_inv = self._check_invariant(
            "MaxComponent",
            lambda s: float(np.max(np.abs(s))),
            trajectory,
        )
        invariants.append(max_inv)

        return invariants

    def _check_invariant(
        self,
        name: str,
        compute: Callable[[NDArray], float],
        trajectory: list[NDArray],
    ) -> DetectedInvariant:
        """Evaluate a candidate invariant over a trajectory."""
        values = [compute(s) for s in trajectory]
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        rel_var = float(std / (abs(mean) + 1e-30))

        return DetectedInvariant(
            name=name,
            compute=compute,
            values=values,
            relative_variation=rel_var,
        )

    def find_attractors(
        self,
        results: list[SimulationResult],
        cluster_tol: float = 0.1,
    ) -> list[DetectedAttractor]:
        """Cluster converged final states into attractors."""
        final_states = [
            r.final_state for r in results if r.converged
        ]
        if not final_states:
            return []

        # Simple clustering by pairwise distance
        attractors: list[DetectedAttractor] = []
        used = set()

        for i, s in enumerate(final_states):
            if i in used:
                continue
            cluster = [s]
            used.add(i)
            for j, t in enumerate(final_states):
                if j in used:
                    continue
                if np.linalg.norm(s - t) < cluster_tol:
                    cluster.append(t)
                    used.add(j)

            attractors.append(DetectedAttractor(
                attractor_type="fixed_point",
                state=np.mean(cluster, axis=0),
                basin_size=len(cluster),
            ))

        return attractors

    def get_conserved(
        self, invariants: list[DetectedInvariant],
    ) -> list[DetectedInvariant]:
        """Filter to well-conserved quantities (relative variation < tolerance)."""
        return [i for i in invariants if i.relative_variation < self.tolerance]

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        invariants: list[DetectedInvariant],
        generation: int = 0,
    ) -> list[int]:
        """Write discovered invariants to knowledge graph."""
        ids = []
        for inv in invariants:
            node = memory.add_node(
                node_type=NodeType.INVARIANT,
                label=inv.name,
                data={
                    "relative_variation": inv.relative_variation,
                    "conserved": inv.relative_variation < self.tolerance,
                },
                confidence=max(0, 1.0 - inv.relative_variation),
                generation=generation,
            )
            ids.append(node.id)
        return ids

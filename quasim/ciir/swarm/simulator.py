r"""Agent SIM — Simulator.

Executes rule sets over state spaces to produce state trajectories
and detect emergent structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quasim.ciir.swarm.memory import (
    KnowledgeGraph,
    NodeType,
)
from quasim.ciir.swarm.physics_lang import PhysicsProgram

# ================================================================
# Simulation result
# ================================================================


@dataclass
class SimulationResult:
    """Result of running a physics program.

    Attributes
    ----------
    trajectory : list of NDArray
        State vectors at each time step.
    converged : bool
        Whether the trajectory converged.
    final_state : NDArray
        Terminal state.
    energy_trace : list of float
        Energy (norm²) at each step.
    """

    trajectory: list[NDArray]
    converged: bool
    final_state: NDArray
    energy_trace: list[float] = field(default_factory=list)


# ================================================================
# Simulator agent
# ================================================================


class Simulator:
    """Execute rule sets over state spaces."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        program: PhysicsProgram,
        initial_state: NDArray,
        steps: int = 100,
        convergence_tol: float = 1e-6,
    ) -> SimulationResult:
        r"""Execute the physics program for *steps* iterations.

        Detects convergence if ||S_{t+1} - S_t|| < tol.
        """
        trajectory = [initial_state.copy()]
        state = initial_state.copy()
        energy_trace = [float(np.sum(state**2))]
        converged = False

        for _ in range(steps):
            new_state = program.execute_step(state)

            if not program.check_consistency(new_state):
                break

            delta = np.linalg.norm(new_state - state)
            state = new_state
            trajectory.append(state.copy())
            energy_trace.append(float(np.sum(state**2)))

            if delta < convergence_tol:
                converged = True
                break

        return SimulationResult(
            trajectory=trajectory,
            converged=converged,
            final_state=state,
            energy_trace=energy_trace,
        )

    def run_ensemble(
        self,
        program: PhysicsProgram,
        n_samples: int = 10,
        steps: int = 100,
    ) -> list[SimulationResult]:
        """Run multiple simulations with random initial conditions."""
        dim = program.state_decl.dim
        results = []
        for _ in range(n_samples):
            s0 = self.rng.standard_normal(dim)
            s0 /= np.linalg.norm(s0)  # Unit sphere
            results.append(self.run(program, s0, steps))
        return results

    def publish_to_memory(
        self,
        memory: KnowledgeGraph,
        result: SimulationResult,
        generation: int = 0,
    ) -> int:
        """Write simulation output to knowledge graph."""
        node = memory.add_node(
            node_type=NodeType.SIMULATION_OUTPUT,
            label=f"Sim_gen{generation}",
            data={
                "trajectory_length": len(result.trajectory),
                "converged": result.converged,
                "final_energy": result.energy_trace[-1] if result.energy_trace else 0,
            },
            generation=generation,
        )
        return node.id

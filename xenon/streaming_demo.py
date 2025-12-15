"""Demonstration of XENON streaming visualization.

This script demonstrates how to integrate XENON simulations with the
streaming visualization pipeline for real-time bio-mechanism visualization.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qubic.visualization.adapters.xenon_adapter import XenonSimulationAdapter
from xenon.core.mechanism import BioMechanism, MolecularState, Transition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSASimulationEngine:
    """Stochastic Simulation Algorithm (SSA) engine for bio-mechanisms.

    Simulates the time evolution of biochemical reaction networks using
    the Gillespie algorithm.
    """

    def __init__(self, mechanism: BioMechanism, initial_state: str, total_time: float = 10.0):
        """Initialize SSA engine.

        Args:
            mechanism: BioMechanism to simulate
            initial_state: ID of initial molecular state
            total_time: Total simulation time in seconds
        """
        self.mechanism = mechanism
        self.current_state = initial_state
        self.current_time = 0.0
        self.total_time = total_time
        self.trajectory = [(0.0, initial_state)]

    def step(self) -> tuple[float, str, str]:
        """Perform one SSA step.

        Returns:
            Tuple of (time_increment, from_state, to_state)
        """
        # Get available transitions from current state
        transitions = self.mechanism.get_transitions_from(self.current_state)

        if not transitions:
            return (float("inf"), self.current_state, self.current_state)

        # Calculate total propensity
        total_rate = sum(t.rate_constant for t in transitions)

        if total_rate == 0:
            return (float("inf"), self.current_state, self.current_state)

        # Sample waiting time (exponential distribution)
        tau = np.random.exponential(1.0 / total_rate)

        # Select which transition occurs (weighted by rate constants)
        rand = np.random.uniform(0, total_rate)
        cumsum = 0.0
        selected_transition = transitions[0]

        for t in transitions:
            cumsum += t.rate_constant
            if rand <= cumsum:
                selected_transition = t
                break

        # Update state
        from_state = self.current_state
        to_state = selected_transition.target_state
        self.current_state = to_state
        self.current_time += tau
        self.trajectory.append((self.current_time, to_state))

        return (tau, from_state, to_state)

    def run(self) -> list[tuple[float, str]]:
        """Run simulation until completion.

        Returns:
            List of (time, state_id) tuples representing the trajectory
        """
        while self.current_time < self.total_time:
            tau, from_state, to_state = self.step()

            if tau == float("inf"):
                break

            logger.debug(f"t={self.current_time:.3f}s: {from_state} -> {to_state} (Δt={tau:.3f}s)")

        return self.trajectory


def create_demo_mechanism() -> BioMechanism:
    """Create a demonstration bio-mechanism.

    Returns:
        Sample BioMechanism with realistic parameters
    """
    # Create a simple protein folding pathway
    states = [
        MolecularState("Unfolded", "ProteinX", 0.0, 1.0),
        MolecularState("Intermediate1", "ProteinX", -15.0, 0.0),
        MolecularState("Intermediate2", "ProteinX", -20.0, 0.0),
        MolecularState("Native", "ProteinX", -35.0, 0.0),
    ]

    transitions = [
        Transition("Unfolded", "Intermediate1", 2.0, -15.0, 25.0),
        Transition("Intermediate1", "Unfolded", 0.5, 15.0, 40.0),
        Transition("Intermediate1", "Intermediate2", 1.5, -5.0, 20.0),
        Transition("Intermediate2", "Intermediate1", 0.3, 5.0, 25.0),
        Transition("Intermediate2", "Native", 3.0, -15.0, 18.0),
        Transition("Native", "Intermediate2", 0.1, 15.0, 33.0),
    ]

    return BioMechanism(
        mechanism_id="ProteinX_Folding",
        states=states,
        transitions=transitions,
        evidence_score=0.92,
    )


def run_streaming_demo(num_steps: int = 10, min_evidence: float = 0.8) -> None:
    """Run streaming visualization demo.

    Args:
        num_steps: Number of simulation steps to visualize
        min_evidence: Minimum evidence threshold for mechanisms
    """
    logger.info("Starting XENON streaming demo")

    # Create demo mechanism
    mechanism = create_demo_mechanism()

    if mechanism.evidence_score < min_evidence:
        logger.warning(
            f"Mechanism evidence {mechanism.evidence_score} below threshold {min_evidence}"
        )
        return

    logger.info(
        f"Created mechanism: {len(mechanism.states)} states, "
        f"{len(mechanism.transitions)} transitions, "
        f"evidence={mechanism.evidence_score:.2f}"
    )

    # Create visualization adapter
    adapter = XenonSimulationAdapter(layout="spring", scale=10.0)

    # Initialize SSA engine
    engine = SSASimulationEngine(mechanism, initial_state="Unfolded", total_time=5.0)

    logger.info("Running SSA simulation...")

    # Simulate and collect snapshots
    snapshots = []
    for i in range(num_steps):
        tau, from_state, to_state = engine.step()

        if tau == float("inf"):
            logger.info("Simulation reached absorbing state")
            break

        logger.info(f"Step {i + 1}: t={engine.current_time:.3f}s, {from_state} -> {to_state}")

        # Create snapshot with updated state concentrations
        snapshot_states = []
        for state in mechanism.states:
            # Update concentration based on current state
            concentration = 1.0 if state.state_id == to_state else 0.0
            snapshot_state = MolecularState(
                state.state_id,
                state.protein_name,
                state.free_energy,
                concentration,
            )
            snapshot_states.append(snapshot_state)

        snapshot_mechanism = BioMechanism(
            mechanism.mechanism_id,
            snapshot_states,
            mechanism.transitions,
            mechanism.evidence_score,
        )
        snapshots.append(snapshot_mechanism)

        # Simulate processing time
        time.sleep(0.1)

    # Convert snapshots to visualization data
    logger.info(f"Converting {len(snapshots)} snapshots to visualization data")
    viz_data_list = adapter.load_mechanism_timeseries(snapshots)

    logger.info(f"Generated {len(viz_data_list)} visualization frames")

    # In a real implementation, these would be streamed to the visualization pipeline
    # For now, just report statistics
    for i, viz_data in enumerate(viz_data_list):
        logger.info(
            f"Frame {i + 1}: {len(viz_data.vertices)} vertices, "
            f"{len(viz_data.faces)} faces, "
            f"fields: {list(viz_data.scalar_fields.keys())}"
        )

    logger.info("Demo complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XENON streaming visualization demo")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of simulation steps (default: 10)",
    )
    parser.add_argument(
        "--min-evidence",
        type=float,
        default=0.8,
        help="Minimum evidence threshold (default: 0.8)",
    )

    args = parser.parse_args()

    run_streaming_demo(num_steps=args.steps, min_evidence=args.min_evidence)

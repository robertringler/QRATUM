"""Command-line interface for XENON simulation and visualization."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from xenon.adapters.bio_mechanism_adapter import BioMechanismAdapter
from xenon.core.mechanism import BioMechanism, MolecularState, Transition

logger = logging.getLogger(__name__)


def create_sample_mechanism(num_states: int = 5) -> BioMechanism:
    """Create a sample bio-mechanism for demonstration.

    Args:
        num_states: Number of states in the mechanism

    Returns:
        Sample BioMechanism instance
    """
    states = []
    for i in range(num_states):
        state = MolecularState(
            state_id=f"S{i}",
            protein_name=f"Protein_{chr(65 + i)}",
            free_energy=np.random.uniform(-50, 0),
            concentration=np.random.uniform(0, 1),
        )
        states.append(state)

    transitions = []
    for i in range(num_states - 1):
        transition = Transition(
            source_state=f"S{i}",
            target_state=f"S{i + 1}",
            rate_constant=np.random.uniform(0.1, 10.0),
            delta_g=np.random.uniform(-10, 10),
            activation_energy=np.random.uniform(10, 50),
        )
        transitions.append(transition)

    # Add some backward transitions
    for i in range(1, num_states, 2):
        transition = Transition(
            source_state=f"S{i}",
            target_state=f"S{max(0, i - 2)}",
            rate_constant=np.random.uniform(0.01, 1.0),
            delta_g=np.random.uniform(-5, 5),
            activation_energy=np.random.uniform(20, 60),
        )
        transitions.append(transition)

    return BioMechanism(
        mechanism_id="SAMPLE_MECH_001",
        states=states,
        transitions=transitions,
        evidence_score=0.85,
    )


def run_simulation(args: argparse.Namespace) -> int:
    """Run XENON simulation.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    logger.info(f"Running XENON simulation with {args.num_states} states")

    # Create sample mechanism
    mechanism = create_sample_mechanism(args.num_states)

    logger.info(
        f"Created mechanism: {len(mechanism.states)} states, "
        f"{len(mechanism.transitions)} transitions"
    )

    # Output mechanism info
    if args.verbose:
        print("\nMechanism States:")
        for state in mechanism.states:
            print(f"  {state.state_id}: {state.protein_name}, ΔG={state.free_energy:.2f} kJ/mol")

        print("\nTransitions:")
        for t in mechanism.transitions:
            print(
                f"  {t.source_state} -> {t.target_state}: "
                f"k={t.rate_constant:.3f} s⁻¹, ΔG={t.delta_g:.2f} kJ/mol"
            )

    # Visualization if requested
    if args.visualize:
        logger.info(f"Starting visualization with backend: {args.viz_backend}")
        visualize_mechanism(mechanism, args)

    return 0


def visualize_mechanism(mechanism: BioMechanism, args: argparse.Namespace) -> None:
    """Visualize bio-mechanism using streaming pipeline.

    Args:
        mechanism: BioMechanism to visualize
        args: Command-line arguments
    """
    # Convert to visualization model
    adapter = BioMechanismAdapter(mechanism)
    viz_model = adapter.to_viz_model()

    print("\nVisualization model created:")
    print(f"  Nodes: {len(viz_model['nodes'])}")
    print(f"  Edges: {len(viz_model['edges'])}")
    print(f"  Evidence score: {viz_model['evidence_score']:.3f}")

    # For now, just print the graph data
    # Full streaming integration would require async event loop
    if args.viz_backend == "matplotlib":
        print("\nNote: Full matplotlib rendering not yet implemented.")
        print("Use --export-json to save visualization data.")

    if args.export_json:
        import json
        import os

        output_file = args.export_json
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        with open(output_file, "w") as f:
            # Convert numpy types to native Python for JSON serialization
            json_data = {
                "nodes": [
                    {k: float(v) if isinstance(v, np.floating) else v for k, v in node.items()}
                    for node in viz_model["nodes"]
                ],
                "edges": [
                    {k: float(v) if isinstance(v, np.floating) else v for k, v in edge.items()}
                    for edge in viz_model["edges"]
                ],
                "evidence_score": float(viz_model["evidence_score"]),
            }
            json.dump(json_data, f, indent=2)

        logger.info(f"Visualization data exported to {output_file}")


def main() -> int:
    """Main entry point for XENON CLI.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="XENON: Bio-mechanism simulation and visualization"
    )

    parser.add_argument(
        "--num-states",
        type=int,
        default=5,
        help="Number of molecular states in the mechanism (default: 5)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of simulation output",
    )

    parser.add_argument(
        "--viz-backend",
        choices=["gpu", "headless", "matplotlib"],
        default="matplotlib",
        help="Visualization backend (default: matplotlib)",
    )

    parser.add_argument(
        "--export-json",
        type=str,
        help="Export visualization data to JSON file",
    )

    parser.add_argument(
        "--min-evidence",
        type=float,
        default=0.8,
        help="Minimum evidence score for mechanisms (default: 0.8)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        return run_simulation(args)
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

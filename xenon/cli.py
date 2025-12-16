"""Command-line interface for XENON.

Commands:
- xenon run: Run XENON learning loop
- xenon query: Query learned mechanisms
- xenon validate: Validate mechanism file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click package not installed. Install with: pip install click")
    sys.exit(1)

from .core.mechanism import BioMechanism
from .runtime.xenon_kernel import XENONRuntime


@click.group()
@click.version_option(version="0.1.0", prog_name="xenon")
def cli():
    """XENON: Xenobiotic Execution Network for Organismal Neurosymbolic reasoning.

    A post-GPU biological intelligence platform for mechanism-based learning.
    """
    pass


@cli.command()
@click.option(
    "--target",
    required=True,
    help="Target protein name",
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    help="Maximum iterations",
)
@click.option(
    "--objective",
    default="characterize",
    help="Learning objective (characterize, find_inhibitor, etc.)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for results (JSON)",
)
def run(
    target: str,
    max_iter: int,
    objective: str,
    seed: Optional[int],
    output: Optional[str],
):
    """Run XENON learning loop.

    Example:
        xenon run --target EGFR --max-iter 50 --objective characterize
    """
    click.echo("🧬 XENON Runtime Starting")
    click.echo(f"Target: {target}")
    click.echo(f"Objective: {objective}")
    click.echo(f"Max iterations: {max_iter}")
    click.echo()

    # Initialize runtime
    runtime = XENONRuntime()
    runtime.add_target(
        name=f"{target}_target",
        protein=target,
        objective=objective,
    )

    # Run learning loop
    click.echo("Running continuous learning loop...")
    summary = runtime.run(max_iterations=max_iter, seed=seed)

    # Display results
    click.echo()
    click.echo("=" * 60)
    click.echo("XENON Learning Summary")
    click.echo("=" * 60)
    click.echo(f"Iterations: {summary['iterations']}")
    click.echo(f"Converged: {summary['converged']}")
    click.echo(f"Final entropy: {summary['final_entropy']:.4f}")
    click.echo(f"Mechanisms discovered: {summary['mechanisms_discovered']}")
    click.echo()

    # Get high-confidence mechanisms
    mechanisms = runtime.get_mechanisms(min_evidence=0.1)

    if mechanisms:
        click.echo(f"Top {min(5, len(mechanisms))} mechanisms:")
        for i, mech in enumerate(mechanisms[:5], 1):
            click.echo(f"  {i}. {mech.name} (posterior: {mech.posterior:.4f})")

    # Save results if output specified
    if output:
        results = {
            "summary": summary,
            "runtime_summary": runtime.get_summary(),
            "mechanisms": [m.to_dict() for m in mechanisms[:10]],
        }

        output_path = Path(output)
        output_path.write_text(json.dumps(results, indent=2))
        click.echo()
        click.echo(f"Results saved to: {output}")


@cli.command()
@click.option(
    "--target",
    required=True,
    help="Target protein name",
)
@click.option(
    "--min-evidence",
    default=0.5,
    type=float,
    help="Minimum posterior probability",
)
@click.option(
    "--input",
    type=click.Path(exists=True),
    default=None,
    help="Input file with XENON results (JSON)",
)
def query(target: str, min_evidence: float, input: Optional[str]):
    """Query learned mechanisms.

    Example:
        xenon query --target EGFR --min-evidence 0.7
    """
    if not input:
        click.echo("Error: --input required for query command")
        click.echo("First run: xenon run --target EGFR --output results.json")
        click.echo("Then query: xenon query --target EGFR --input results.json")
        return

    # Load results
    input_path = Path(input)
    data = json.loads(input_path.read_text())

    # Filter mechanisms
    mechanisms_data = data.get("mechanisms", [])
    filtered = [m for m in mechanisms_data if m.get("posterior", 0) >= min_evidence]

    click.echo(f"🔍 Query Results for {target}")
    click.echo(f"Minimum evidence: {min_evidence}")
    click.echo(f"Mechanisms found: {len(filtered)}")
    click.echo()

    for i, mech_data in enumerate(filtered, 1):
        click.echo(f"{i}. {mech_data['name']}")
        click.echo(f"   Posterior: {mech_data['posterior']:.4f}")
        click.echo(f"   States: {len(mech_data['states'])}")
        click.echo(f"   Transitions: {len(mech_data['transitions'])}")
        click.echo(f"   Hash: {mech_data['hash'][:16]}...")
        click.echo()


@cli.command()
@click.option(
    "--mechanism-file",
    required=True,
    type=click.Path(exists=True),
    help="Mechanism file (JSON)",
)
@click.option(
    "--temperature",
    default=310.0,
    type=float,
    help="Temperature in Kelvin",
)
def validate(mechanism_file: str, temperature: float):
    """Validate mechanism file.

    Example:
        xenon validate --mechanism-file mechanism.json
    """
    click.echo(f"🔬 Validating mechanism: {mechanism_file}")
    click.echo()

    # Load mechanism
    mechanism_path = Path(mechanism_file)
    data = json.loads(mechanism_path.read_text())

    try:
        mechanism = BioMechanism.from_dict(data)
    except Exception as e:
        click.echo(f"❌ Error loading mechanism: {e}")
        return

    # Validate
    click.echo(f"Mechanism: {mechanism.name}")
    click.echo(f"States: {len(mechanism._states)}")
    click.echo(f"Transitions: {len(mechanism._transitions)}")
    click.echo()

    # Thermodynamic feasibility
    is_feasible = mechanism.is_thermodynamically_feasible(temperature)
    if is_feasible:
        click.echo("✅ Thermodynamically feasible")
    else:
        click.echo("❌ Thermodynamically infeasible")

    # Conservation laws
    is_valid, violations = mechanism.validate_conservation_laws()
    if is_valid:
        click.echo("✅ Conservation laws satisfied")
    else:
        click.echo("❌ Conservation law violations:")
        for violation in violations:
            click.echo(f"   - {violation}")

    # Mechanism hash
    mech_hash = mechanism.compute_mechanism_hash()
    click.echo()
    click.echo(f"Mechanism hash: {mech_hash[:32]}...")


if __name__ == "__main__":
    cli()
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

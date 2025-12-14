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
    click.echo(f"🧬 XENON Runtime Starting")
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
    filtered = [
        m for m in mechanisms_data
        if m.get("posterior", 0) >= min_evidence
    ]
    
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

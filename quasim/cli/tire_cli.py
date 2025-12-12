"""CLI tool for tire simulation library generation."""

import click

from quasim.domains.tire import generate_tire_library

try:
    from integrations.goodyear import create_goodyear_library
    GOODYEAR_AVAILABLE = True
except ImportError:
    GOODYEAR_AVAILABLE = False


@click.group()
def cli():
    """QuASIM Tire Simulation Library - Goodyear Quantum Pilot Platform."""
    pass


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="tire_simulation_library",
    help="Output directory for simulation library",
)
@click.option(
    "--count",
    "-n",
    default=10000,
    type=int,
    help="Number of simulation scenarios to generate",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv", "both"]),
    default="both",
    help="Export format",
)
@click.option(
    "--run/--no-run",
    default=True,
    help="Run simulations or just generate scenarios",
)
def generate(output_dir: str, count: int, format: str, run: bool):
    """Generate comprehensive tire simulation library."""
    click.echo(f"Generating tire simulation library with {count} scenarios...")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Export format: {format}")
    click.echo(f"Run simulations: {run}")
    click.echo("")

    summary = generate_tire_library(
        output_dir=output_dir,
        scenario_count=count,
        run_simulations=run,
        export_format=format,
    )

    click.echo("")
    click.echo("=" * 60)
    click.echo("Library Generation Complete")
    click.echo("=" * 60)
    click.echo(f"Total scenarios: {summary['total_scenarios']}")
    click.echo(f"Output directory: {summary['output_directory']}")

    if "statistics" in summary:
        click.echo("")
        click.echo("Performance Statistics:")
        stats = summary["statistics"]
        click.echo(f"  Average Grip: {stats.get('avg_grip_coefficient', 'N/A')}")
        click.echo(f"  Average Rolling Resistance: {stats.get('avg_rolling_resistance', 'N/A')}")
        click.echo(
            f"  Average Optimization Score: {stats.get('avg_optimization_score', 'N/A')}"
        )


@cli.command()
@click.argument("simulation_id")
@click.option(
    "--library-dir",
    "-l",
    default="tire_simulation_library",
    help="Library directory",
)
def inspect(simulation_id: str, library_dir: str):
    """Inspect a specific simulation result."""
    import json
    from pathlib import Path

    json_file = Path(library_dir) / "tire_simulation_results.json"

    if not json_file.exists():
        click.echo(f"Error: Library not found at {json_file}")
        return

    with open(json_file) as f:
        results = json.load(f)

    # Find simulation
    result = None
    for r in results:
        if r["simulation_id"] == simulation_id:
            result = r
            break

    if not result:
        click.echo(f"Error: Simulation {simulation_id} not found")
        return

    click.echo("=" * 60)
    click.echo(f"Simulation: {simulation_id}")
    click.echo("=" * 60)
    click.echo("")

    # Display input parameters
    click.echo("Input Parameters:")
    params = result["input_parameters"]
    click.echo(f"  Tire Type: {params['tire']['tire_type']}")
    click.echo(f"  Compound: {params['compound']['compound_type']}")
    click.echo(f"  Environment: {params['environment']['surface_type']}")
    click.echo(f"  Temperature: {params['environment']['ambient_temperature']}°C")
    click.echo(f"  Load: {params['load_kg']} kg")
    click.echo(f"  Pressure: {params['pressure_kpa']} kPa")
    click.echo(f"  Speed: {params['speed_kmh']} km/h")
    click.echo("")

    # Display performance metrics
    click.echo("Performance Metrics:")
    metrics = result["performance_metrics"]
    click.echo(f"  Grip Coefficient: {metrics['grip_coefficient']}")
    click.echo(f"  Rolling Resistance: {metrics['rolling_resistance']}")
    click.echo(f"  Wear Rate: {metrics['wear_rate']} mm/1000km")
    click.echo(f"  Thermal Performance: {metrics['thermal_performance']}")
    click.echo(f"  Predicted Lifetime: {metrics['predicted_lifetime_km']} km")
    click.echo(f"  Optimization Score: {metrics['optimization_score']}")
    click.echo("")

    # Display optimization suggestions
    if result.get("optimization_suggestions"):
        click.echo("Optimization Suggestions:")
        for suggestion in result["optimization_suggestions"]:
            click.echo(f"  - {suggestion}")


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="goodyear_tire_library",
    help="Output directory for Goodyear library",
)
@click.option(
    "--scenarios-per-material",
    "-s",
    default=10,
    type=int,
    help="Number of scenarios per material",
)
@click.option(
    "--use-all/--use-certified",
    default=False,
    help="Use all materials or only certified ones",
)
@click.option(
    "--quantum-only",
    is_flag=True,
    help="Use only quantum-validated materials",
)
def goodyear(
    output_dir: str, scenarios_per_material: int, use_all: bool, quantum_only: bool
):
    """Generate library using Goodyear Quantum Pilot materials (1,000+ compounds)."""
    if not GOODYEAR_AVAILABLE:
        click.echo("Error: Goodyear integration not available")
        return

    click.echo("=" * 60)
    click.echo("Goodyear Quantum Pilot - QuASIM Integration")
    click.echo("=" * 60)
    click.echo("")

    filters = {}
    if quantum_only:
        filters["quantum_validated"] = True
        click.echo("Using quantum-validated materials only")
    elif not use_all:
        filters["certification_status"] = "certified"
        click.echo("Using certified materials only")
    else:
        click.echo("Using all 1,000+ Goodyear materials")

    click.echo(f"Scenarios per material: {scenarios_per_material}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("")

    summary = create_goodyear_library(
        output_dir=output_dir,
        scenarios_per_material=scenarios_per_material,
        use_all_materials=use_all,
        material_filters=filters if filters else None,
    )

    click.echo("")
    click.echo("=" * 60)
    click.echo("Goodyear Library Generation Complete")
    click.echo("=" * 60)
    click.echo(f"Materials used: {summary['total_materials']}")
    click.echo(f"Total scenarios: {summary['total_scenarios']}")
    click.echo(f"Output directory: {summary['output_directory']}")
    click.echo("")

    if "statistics" in summary:
        click.echo("Performance Statistics:")
        stats = summary["statistics"]
        click.echo(f"  Average Grip: {stats.get('avg_grip_coefficient', 'N/A')}")
        click.echo(f"  Average Rolling Resistance: {stats.get('avg_rolling_resistance', 'N/A')}")
        click.echo(
            f"  Average Optimization Score: {stats.get('avg_optimization_score', 'N/A')}"
        )

    if "materials_database_stats" in summary:
        click.echo("")
        click.echo("Materials Database:")
        db_stats = summary["materials_database_stats"]
        click.echo(f"  Total Materials: {db_stats['total_materials']}")
        click.echo(f"  Quantum Validated: {db_stats['quantum_validated']} ({db_stats['quantum_validated_percentage']}%)")


if __name__ == "__main__":
    cli()

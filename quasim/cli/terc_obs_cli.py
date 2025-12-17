#!/usr/bin/env python3
"""QuASIM TERC-OBS CLI - TERC observable emission."""

import json
import sys

import click
import numpy as np

from quasim.terc_bridge.adapters import from_quasim_state, to_terc_observable_format
from quasim.terc_bridge.observables import (
    beta_metrics_from_cipher,
    emergent_complexity,
    ioc_period_candidates,
    qgh_consensus_status,
)
from quasim.terc_bridge.registry import list_observables


@click.group()
@click.version_option(version="0.1.0", prog_name="quasim-terc-obs")
def cli():
    """QuASIM TERC Observable Emission Tool.

    Extract and emit observables from REVULTRA/QGH algorithms for
    TERC validation tiers.
    """

    pass


@cli.command("emit")
@click.option("--state-file", type=click.Path(exists=True), help="Path to QuASIM state file")
@click.option("--text", type=str, help="Text/ciphertext input")
@click.option("--out", type=click.Path(writable=True), required=True, help="Output JSON file")
@click.option("--observable", type=str, help="Specific observable to compute")
def emit(state_file: str | None, text: str | None, out: str, observable: str | None):
    """Emit TERC observables from QuASIM state or text input.

    Examples:
        quasim-terc-obs emit --text "ATTACK" --out observables.json
        quasim-terc-obs emit --state-file state.json --out obs.json --observable beta_metrics
    """

    # Load input
    if state_file:
        state_input = from_quasim_state(state_file)
    elif text:
        state_input = text
    else:
        click.echo("Error: Must specify either --state-file or --text", err=True)
        sys.exit(1)

    results = {}

    if observable:
        # Compute specific observable
        try:
            if observable == "beta_metrics":
                results[observable] = beta_metrics_from_cipher(str(state_input))
            elif observable == "ioc_periods":
                results[observable] = ioc_period_candidates(str(state_input))
            elif observable == "emergent_complexity":
                results[observable] = emergent_complexity(str(state_input))
            else:
                click.echo(f"Unknown observable: {observable}", err=True)
                click.echo(f"Available: {list_observables()}", err=True)
                sys.exit(1)
        except Exception as e:
            click.echo(f"Error computing observable: {e}", err=True)
            sys.exit(1)
    else:
        # Compute all text-based observables
        if isinstance(state_input, str):
            click.echo("Computing REVULTRA observables...")
            results["beta_metrics"] = beta_metrics_from_cipher(state_input)
            results["ioc_periods"] = ioc_period_candidates(state_input)
            results["emergent_complexity"] = emergent_complexity(state_input)
        else:
            click.echo("Error: Non-text input requires --observable specification", err=True)
            sys.exit(1)

    # Format for TERC
    formatted = to_terc_observable_format(results)

    # Write output
    with open(out, "w") as f:
        json.dump(formatted, f, indent=2)

    click.echo(f"Observables emitted to: {out}")
    click.echo(json.dumps(formatted, indent=2))


@cli.command("list")
def list_cmd():
    """List all registered observables."""

    observables = list_observables()

    click.echo("Registered TERC Observables:")
    click.echo("=" * 40)
    for obs in observables:
        click.echo(f"  - {obs}")
    click.echo(f"\nTotal: {len(observables)} observables")


@cli.command("consensus")
@click.option("--num-nodes", type=int, default=5, help="Number of nodes")
@click.option("--state-dim", type=int, default=3, help="State dimension")
@click.option("--out", type=click.Path(writable=True), help="Output JSON file")
def consensus(num_nodes: int, state_dim: int, out: str | None):
    """Compute consensus status observable for QGH.

    Example:
        quasim-terc-obs consensus --num-nodes 10 --state-dim 5 --out consensus.json
    """

    click.echo(f"Computing consensus for {num_nodes} nodes, dimension {state_dim}...")

    # Generate random initial states
    rng = np.random.default_rng(42)
    states = rng.normal(0, 1, size=(num_nodes, state_dim))

    # Compute consensus
    status = qgh_consensus_status(states)

    formatted = to_terc_observable_format({"consensus_status": status})

    if out:
        with open(out, "w") as f:
            json.dump(formatted, f, indent=2)
        click.echo(f"Results written to: {out}")
    else:
        click.echo(json.dumps(formatted, indent=2))

    click.echo(f"\nSummary: Converged={status['converged']}, Stability={status['stability']:.3f}")


@cli.command("validate")
@click.option(
    "--obs-file", type=click.Path(exists=True), required=True, help="Observable JSON file"
)
def validate(obs_file: str):
    """Validate observable JSON format.

    Example:
        quasim-terc-obs validate --obs-file observables.json
    """

    try:
        with open(obs_file) as f:
            data = json.load(f)

        click.echo("Validating observable file...")

        # Check required fields
        if "observables" not in data:
            click.echo("❌ Missing 'observables' field", err=True)
            sys.exit(1)

        if "format_version" not in data:
            click.echo("⚠️  Warning: Missing 'format_version' field")

        if "source" not in data:
            click.echo("⚠️  Warning: Missing 'source' field")

        click.echo("✓ Observable file is valid")
        click.echo(f"Format version: {data.get('format_version', 'unknown')}")
        click.echo(f"Source: {data.get('source', 'unknown')}")
        click.echo(f"Observable count: {len(data['observables'])}")

        # List observables
        click.echo("\nObservables present:")
        for key in data["observables"]:
            click.echo(f"  - {key}")

    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Validation error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

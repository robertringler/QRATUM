#!/usr/bin/env python3
"""QuASIM REVULTRA CLI - Cryptanalytic algorithms."""

import json
import sys

import click

from quasim.revultra.algorithms import REVULTRAAlgorithms
from quasim.revultra.demos import KRYPTOS_K4, demo_kryptos_k4_analysis
from quasim.revultra.io import export_to_json, load_ciphertext
from quasim.revultra.metrics import find_peaks


@click.group()
@click.version_option(version="0.1.0", prog_name="quasim-revultra")
def cli():
    """QuASIM REVULTRA - Cryptanalytic Analysis Tools.

    Quantum-inspired algorithms for ciphertext analysis including
    holographic entropy, temporal embeddings, and spectral analysis.
    """

    pass


@cli.command("analyze")
@click.option("--ciphertext", type=str, help="Ciphertext string to analyze")
@click.option("--file", type=click.Path(exists=True), help="Path to ciphertext file")
@click.option("--plot", is_flag=True, help="Generate plots (requires matplotlib)")
@click.option("--export", type=click.Path(writable=True), help="Export results to JSON file")
@click.option("--max-period", type=int, default=20, help="Maximum period for IoC analysis")
def analyze(
    ciphertext: str | None, file: str | None, plot: bool, export: str | None, max_period: int
):
    """Run comprehensive REVULTRA analysis on ciphertext.

    Performs quantum information topology, holographic entropy transform,
    temporal embeddings, pattern mining, IoC analysis, spectral autocorrelation,
    and emergent complexity scoring.

    Example:
        quasim-revultra analyze --ciphertext "ATTACKATDAWN" --export results.json
    """

    # Load ciphertext
    if file:
        text = load_ciphertext(file)
    elif ciphertext:
        text = ciphertext
    else:
        click.echo("Error: Must specify either --ciphertext or --file", err=True)
        sys.exit(1)

    click.echo(f"Analyzing ciphertext (length: {len(text)})...")

    # Run analysis
    rev = REVULTRAAlgorithms()

    results = {
        "input_length": len(text),
        "frequency_analysis": rev.frequency_analysis(text),
        "complexity": rev.emergent_complexity_score(text),
        "chi_squared": rev.chi_squared_test(text),
        "patterns": rev.memory_recursive_pattern_mining(text),
    }

    # IoC analysis with peak detection
    ioc = rev.index_of_coincidence_tensor(text, max_period=max_period)
    ioc_peaks = find_peaks(ioc, threshold=0.2)
    results["ioc_tensor"] = ioc
    results["ioc_peaks"] = ioc_peaks

    # Autocorrelation
    autocorr = rev.spectral_autocorrelation(text, max_lag=max_period)
    autocorr_peaks = find_peaks(autocorr, threshold=0.3)
    results["autocorrelation"] = autocorr
    results["autocorr_peaks"] = autocorr_peaks

    # Holographic analysis
    entropy, surface = rev.holographic_entropy_transform(text)
    results["holographic_entropy"] = entropy
    results["surface_shape"] = surface.shape

    # Quantum topology
    topology = rev.quantum_information_topology(text)
    results["topology_size"] = len(topology)

    # Output results
    if export:
        export_to_json(results, export)
        click.echo(f"Results exported to: {export}")
    else:
        # Pretty print to stdout
        output = json.dumps(
            results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        )
        click.echo(output)

    # Summary
    click.echo("\n--- Summary ---")
    click.echo(f"Complexity Score: {results['complexity']['score']:.2f}")
    click.echo(f"Chi-squared: {results['chi_squared']:.2f}")
    click.echo(f"IoC Peaks at periods: {ioc_peaks}")
    click.echo(f"Pattern count: {len(results['patterns'])}")

    if plot:
        click.echo("\nNote: Plotting requires matplotlib (not implemented in this version)")


@cli.command("demo")
@click.option(
    "--section",
    type=click.Choice(["kryptos", "simple", "all"]),
    default="kryptos",
    help="Demo section to run",
)
@click.option("--export", type=click.Path(writable=True), help="Export demo results to JSON")
def demo(section: str, export: str | None):
    """Run REVULTRA demonstration examples.

    Examples:
        quasim-revultra demo --section kryptos
        quasim-revultra demo --section all --export demo_results.json
    """

    if section in ["kryptos", "all"]:
        click.echo("=" * 70)
        click.echo("REVULTRA Kryptos K4 Analysis Demo")
        click.echo("=" * 70)
        click.echo(f"Analyzing: {KRYPTOS_K4}")

        results = demo_kryptos_k4_analysis()

        click.echo(f"\nComplexity Score: {results['complexity']['score']:.2f}")
        click.echo(f"Entropy: {results['complexity']['entropy']:.4f}")
        click.echo(f"IoC Peaks: {results.get('ioc_peaks', [])}")
        click.echo(f"Patterns detected: {len(results['patterns'])}")

        if export:
            export_to_json({"kryptos_k4": results}, export)
            click.echo(f"\nResults exported to: {export}")

    if section == "simple":
        click.echo("Running simple cipher demo...")
        click.echo("(Simple demo not implemented - use 'analyze' command)")


@cli.command("frequency")
@click.option("--ciphertext", type=str, help="Ciphertext to analyze")
@click.option("--file", type=click.Path(exists=True), help="Path to ciphertext file")
def frequency(ciphertext: str | None, file: str | None):
    """Perform frequency analysis on ciphertext.

    Example:
        quasim-revultra frequency --ciphertext "HELLO WORLD"
    """

    if file:
        text = load_ciphertext(file)
    elif ciphertext:
        text = ciphertext
    else:
        click.echo("Error: Must specify either --ciphertext or --file", err=True)
        sys.exit(1)

    rev = REVULTRAAlgorithms()
    freqs = rev.frequency_analysis(text)

    click.echo("Character Frequencies:")
    click.echo("-" * 40)
    for char in sorted(freqs.keys()):
        click.echo(f"{char}: {freqs[char]:6.2f}%")


if __name__ == "__main__":
    cli()

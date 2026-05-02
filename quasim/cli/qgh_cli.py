#!/usr/bin/env python3
"""QuASIM QGH CLI - Non-speculative quantum algorithms."""

import json
import sys

import click
import numpy as np

from quasim.qgh.demos import (demo_av_fusion, demo_distributed_monitoring,
                              demo_federated_consensus,
                              demo_portfolio_optimization,
                              demo_signal_processing, demo_tensor_processing)
from quasim.qgh.io import export_results_to_json


@click.group()
@click.version_option(version="0.1.0", prog_name="quasim-qgh")
def cli():
    """QuASIM QGH - Non-Speculative Quantum Algorithms.

    Distributed quantum system algorithms including causal history,
    superposition resolution, stream monitoring, and consensus propagation.
    """

    pass


@cli.command("demo")
@click.option(
    "--section",
    type=click.Choice(
        ["tensor", "distributed", "fusion", "optimization", "signal", "consensus", "all"]
    ),
    default="all",
    help="Demo section to run",
)
@click.option("--export", type=click.Path(writable=True), help="Export demo results to JSON")
def demo(section: str, export: str | None):
    """Run QGH demonstration examples.

    Available demos:
    - tensor: Tensor processing with SuperpositionResolver
    - distributed: Distributed stream monitoring
    - fusion: AV sensor fusion
    - optimization: Portfolio optimization with stability monitoring
    - signal: Signal processing with causal history
    - consensus: Federated consensus propagation
    - all: Run all demos

    Examples:
        quasim-qgh demo --section tensor
        quasim-qgh demo --section all --export qgh_results.json
    """

    demos_to_run = []

    if section == "all":
        demos_to_run = [
            ("Tensor Processing", demo_tensor_processing),
            ("Distributed Monitoring", demo_distributed_monitoring),
            ("Federated Consensus", demo_federated_consensus),
            ("AV Sensor Fusion", demo_av_fusion),
            ("Portfolio Optimization", demo_portfolio_optimization),
            ("Signal Processing", demo_signal_processing),
        ]
    else:
        demo_map = {
            "tensor": ("Tensor Processing", demo_tensor_processing),
            "distributed": ("Distributed Monitoring", demo_distributed_monitoring),
            "fusion": ("AV Sensor Fusion", demo_av_fusion),
            "optimization": ("Portfolio Optimization", demo_portfolio_optimization),
            "signal": ("Signal Processing", demo_signal_processing),
            "consensus": ("Federated Consensus", demo_federated_consensus),
        }
        if section in demo_map:
            demos_to_run = [demo_map[section]]

    all_results = {}

    for name, demo_fn in demos_to_run:
        click.echo("=" * 70)
        click.echo(f"QGH Demo: {name}")
        click.echo("=" * 70)

        result = demo_fn()
        all_results[name.lower().replace(" ", "_")] = result

        # Pretty print key results
        click.echo(json.dumps(result, indent=2, default=str))
        click.echo()

    if export:
        export_results_to_json(all_results, export)
        click.echo(f"All results exported to: {export}")


@cli.command("resolve")
@click.option("--state", type=str, required=True, help="Initial state as comma-separated floats")
@click.option("--iterations", type=int, default=100, help="Maximum iterations")
@click.option("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
def resolve(state: str, iterations: int, tolerance: float):
    """Resolve quantum superposition to consistent state.

    Example:
        quasim-qgh resolve --state "0.5,0.3,0.15,0.05" --iterations 50
    """

    from quasim.qgh.nonspec_algorithms import SuperpositionResolver

    try:
        state_vec = np.array([float(x.strip()) for x in state.split(",")], dtype=np.float64)
    except ValueError:
        click.echo("Error: Invalid state format. Use comma-separated floats.", err=True)
        sys.exit(1)

    click.echo(f"Initial state: {state_vec}")

    # Normalize function
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.sum(x)

    resolver = SuperpositionResolver(max_iterations=iterations, tolerance=tolerance)
    result = resolver.resolve(state_vec, normalize)

    click.echo(f"\nFinal state: {result['state']}")
    click.echo(f"Converged: {result['converged']}")
    click.echo(f"Iterations: {result['iterations']}")
    click.echo(f"Final delta: {result['final_delta']:.2e}")


@cli.command("monitor")
@click.option("--num-streams", type=int, default=3, help="Number of streams to monitor")
@click.option("--samples", type=int, default=50, help="Number of samples per stream")
@click.option("--threshold", type=float, default=0.7, help="Synchronization threshold")
def monitor(num_streams: int, samples: int, threshold: float):
    """Monitor distributed streams and detect synchronization.

    Example:
        quasim-qgh monitor --num-streams 4 --samples 100
    """

    from quasim.qgh.nonspec_algorithms import DistributedStreamMonitor

    click.echo(f"Monitoring {num_streams} streams with {samples} samples each...")

    monitor_obj = DistributedStreamMonitor(num_streams=num_streams, buffer_size=1000)
    rng = np.random.default_rng(42)

    # Simulate correlated streams
    for i in range(samples):
        base = rng.normal(0, 1)
        for stream_id in range(num_streams):
            if stream_id < num_streams // 2:
                # First half: correlated
                value = base + rng.normal(0, 0.1)
            else:
                # Second half: independent
                value = rng.normal(0, 1)
            monitor_obj.add_sample(stream_id, value)

    # Detect synchronization
    sync_pairs = monitor_obj.detect_sync_patterns(threshold=threshold)

    click.echo(f"\nDetected {len(sync_pairs)} synchronized pairs:")
    for i, j in sync_pairs:
        click.echo(f"  Stream {i} <-> Stream {j}")

    # Show statistics
    click.echo("\nStream Statistics:")
    for stream_id in range(num_streams):
        stats = monitor_obj.get_stream_stats(stream_id)
        click.echo(f"  Stream {stream_id}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")


if __name__ == "__main__":
    cli()

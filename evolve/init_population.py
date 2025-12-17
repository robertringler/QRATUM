"""Initialize the first evolving kernel population for Phase III."""

from __future__ import annotations

import argparse
from pathlib import Path

from evolve.introspection import IntrospectionAgent, KernelMetrics
from evolve.rl_controller import RLController


def generate_initial_population(population_size: int = 20, seed: int = 42) -> None:
    """Generate and save initial kernel population."""

    print(f"Initializing kernel population with {population_size} genomes...")

    # Create RL controller
    controller = RLController(population_size=population_size, seed=seed)

    # Generate initial population
    population = controller.initialize_population()

    # Save to disk
    controller.save_policy()

    print(f"Generated {len(population)} initial genomes")
    print("Saved policy to evolve/policies/policy.json")

    # Generate sample introspection data for testing
    agent = IntrospectionAgent()

    for i, genome in enumerate(population[:5]):  # Sample first 5
        metrics = KernelMetrics(
            kernel_id=f"kernel_{i}",
            tile_size=genome.tile_size,
            warp_count=genome.warp_count,
            unroll_factor=genome.unroll_factor,
            async_depth=genome.async_depth,
            precision=genome.precision,
            latency_ms=100.0 + i * 10,
            energy_joules=10.0 + i * 0.5,
            compute_utilization=0.7 + i * 0.02,
        )
        agent.record_metrics(metrics)

    log_path = agent.flush_to_disk()
    print(f"Generated sample metrics at {log_path}")

    # Print population summary
    print("\nInitial Population Summary:")
    print("=" * 60)
    for i, genome in enumerate(population[:10]):  # Show first 10
        print(
            f"Genome {i}: tile={genome.tile_size}, warp={genome.warp_count}, "
            f"unroll={genome.unroll_factor}, async={genome.async_depth}, "
            f"precision={genome.precision}"
        )

    if len(population) > 10:
        print(f"... and {len(population) - 10} more genomes")

    # Generate README
    readme_path = Path("evolve/README.md")
    readme_content = """# Phase III Kernel Evolution

This directory contains the autonomous kernel evolution system for QuASIM.

## Components

- `introspection.py`: Runtime introspection agent for collecting kernel metrics
- `rl_controller.py`: Reinforcement learning controller for kernel optimization
- `init_population.py`: Initial population generator
- `policies/`: Serialized RL policies and kernel genomes
- `logs/`: Introspection logs and telemetry data

## Usage

Initialize population:
```bash
python -m evolve.init_population --population-size 20 --seed 42
```

## Evolution Strategy

The system uses an evolutionary algorithm with:
- Population-based optimization
- Elite selection (top 50%)
- Mutation-based exploration
- Fitness evaluation based on latency and energy

## Kernel Genome

Each kernel genome encodes:
- `tile_size`: Computation tile size (64-1024)
- `warp_count`: Number of warps (8-64)
- `unroll_factor`: Loop unroll factor (1-16)
- `async_depth`: Async pipeline depth (1-8)
- `precision`: Numerical precision (fp8, fp16, bf16, fp32)
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"\nCreated documentation at {readme_path}")
    print("\nInitialization complete!")


def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Initialize kernel evolution population for Phase III"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Size of initial population",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    generate_initial_population(args.population_size, args.seed)


if __name__ == "__main__":
    main()

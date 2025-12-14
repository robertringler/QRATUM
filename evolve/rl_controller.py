"""Reinforcement learning controller for kernel parameter optimization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelGenome:
    """Kernel configuration genome for evolution."""

    tile_size: int = 256
    warp_count: int = 32
    unroll_factor: int = 4
    async_depth: int = 2
    precision: str = "fp32"
    fitness: float = 0.0
    generation: int = 0

    def to_dict(self) -> dict:
        """Convert genome to dictionary."""
        return {
            "tile_size": self.tile_size,
            "warp_count": self.warp_count,
            "unroll_factor": self.unroll_factor,
            "async_depth": self.async_depth,
            "precision": self.precision,
            "fitness": self.fitness,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KernelGenome:
        """Create genome from dictionary."""
        return cls(**data)

    def mutate(self, mutation_rate: float = 0.1) -> KernelGenome:
        """Create a mutated copy of this genome."""
        new_genome = KernelGenome(
            tile_size=self.tile_size,
            warp_count=self.warp_count,
            unroll_factor=self.unroll_factor,
            async_depth=self.async_depth,
            precision=self.precision,
            fitness=0.0,
            generation=self.generation + 1,
        )

        if random.random() < mutation_rate:
            # Mutate tile_size
            new_genome.tile_size = max(64, min(1024, self.tile_size + random.randint(-64, 64)))

        if random.random() < mutation_rate:
            # Mutate warp_count
            new_genome.warp_count = max(8, min(64, self.warp_count + random.randint(-8, 8)))

        if random.random() < mutation_rate:
            # Mutate unroll_factor
            new_genome.unroll_factor = max(1, min(16, self.unroll_factor + random.randint(-2, 2)))

        if random.random() < mutation_rate:
            # Mutate async_depth
            new_genome.async_depth = max(1, min(8, self.async_depth + random.randint(-1, 1)))

        if random.random() < mutation_rate * 0.5:
            # Mutate precision (less frequently)
            precisions = ["fp8", "fp16", "bf16", "fp32"]
            new_genome.precision = random.choice(precisions)

        return new_genome


class RLController:
    """Simplified RL controller using evolutionary strategies for kernel optimization."""

    def __init__(self, population_size: int = 20, seed: int = 42):
        self.population_size = population_size
        self.population: list[KernelGenome] = []
        self.best_genome: KernelGenome | None = None
        self.generation = 0
        random.seed(seed)

    def initialize_population(self) -> list[KernelGenome]:
        """Create initial random population."""
        self.population = []
        tile_sizes = [64, 128, 256, 512, 1024]
        warp_counts = [8, 16, 32, 64]
        unroll_factors = [1, 2, 4, 8, 16]
        async_depths = [1, 2, 4, 8]
        precisions = ["fp8", "fp16", "bf16", "fp32"]

        for _ in range(self.population_size):
            genome = KernelGenome(
                tile_size=random.choice(tile_sizes),
                warp_count=random.choice(warp_counts),
                unroll_factor=random.choice(unroll_factors),
                async_depth=random.choice(async_depths),
                precision=random.choice(precisions),
                generation=0,
            )
            self.population.append(genome)

        return self.population

    def evaluate_fitness(self, genome: KernelGenome, latency_ms: float, energy_j: float) -> float:
        """
        Compute fitness score for a genome based on performance metrics.
        Lower is better (minimizing combined latency and energy).
        """
        # Fitness function: weighted combination of latency and energy
        # Normalize by typical values
        normalized_latency = latency_ms / 100.0  # Assume 100ms baseline
        normalized_energy = energy_j / 10.0  # Assume 10J baseline

        # Combined objective with equal weights
        fitness = normalized_latency + normalized_energy

        genome.fitness = fitness
        return fitness

    def select_and_evolve(self) -> list[KernelGenome]:
        """Select top performers and create next generation."""
        # Sort by fitness (lower is better)
        self.population.sort(key=lambda g: g.fitness)

        # Update best genome
        if self.best_genome is None or self.population[0].fitness < self.best_genome.fitness:
            self.best_genome = KernelGenome(
                tile_size=self.population[0].tile_size,
                warp_count=self.population[0].warp_count,
                unroll_factor=self.population[0].unroll_factor,
                async_depth=self.population[0].async_depth,
                precision=self.population[0].precision,
                fitness=self.population[0].fitness,
                generation=self.generation,
            )

        # Keep top 50% as elites
        elite_count = self.population_size // 2
        elites = self.population[:elite_count]

        # Generate new population
        new_population = list(elites)

        # Fill rest with mutations of elites
        while len(new_population) < self.population_size:
            parent = random.choice(elites)
            child = parent.mutate(mutation_rate=0.2)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return self.population

    def save_policy(self, path: str = "evolve/policies/policy.json") -> None:
        """Save current policy (population and best genome) to disk."""
        policy_path = Path(path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "generation": self.generation,
            "population": [g.to_dict() for g in self.population],
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
        }

        with open(policy_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_policy(self, path: str = "evolve/policies/policy.json") -> None:
        """Load policy from disk."""
        policy_path = Path(path)
        if not policy_path.exists():
            return

        with open(policy_path) as f:
            data = json.load(f)

        self.generation = data["generation"]
        self.population = [KernelGenome.from_dict(g) for g in data["population"]]
        if data["best_genome"]:
            self.best_genome = KernelGenome.from_dict(data["best_genome"])

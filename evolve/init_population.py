"""Initialize the first evolving kernel population for Phase III autonomous evolution.

This module creates the initial genome pool for reinforcement-learning-driven
kernel optimization. Each genome encodes kernel parameters like tile size,
warp count, unroll factors, and async depth.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class KernelGenome:
    """Represents a kernel configuration genome for evolutionary optimization."""
    
    genome_id: str
    tile_size: int
    warp_count: int
    unroll_factor: int
    async_depth: int
    precision: str
    timestamp: float
    generation: int = 0
    fitness: float = 0.0
    
    def mutate(self, mutation_rate: float = 0.1) -> KernelGenome:
        """Create a mutated copy of this genome."""
        new_genome = KernelGenome(
            genome_id=f"{self.genome_id}_m{int(time.time())}",
            tile_size=self._mutate_value(self.tile_size, 8, 128, mutation_rate),
            warp_count=self._mutate_value(self.warp_count, 1, 32, mutation_rate),
            unroll_factor=self._mutate_value(self.unroll_factor, 1, 16, mutation_rate),
            async_depth=self._mutate_value(self.async_depth, 1, 8, mutation_rate),
            precision=self.precision,
            timestamp=time.time(),
            generation=self.generation + 1,
            fitness=0.0
        )
        return new_genome
    
    @staticmethod
    def _mutate_value(value: int, min_val: int, max_val: int, rate: float) -> int:
        """Mutate an integer value within bounds."""
        if random.random() < rate:
            delta = random.randint(-2, 2)
            return max(min_val, min(max_val, value + delta))
        return value
    
    def to_dict(self) -> dict:
        """Convert genome to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> KernelGenome:
        """Create genome from dictionary."""
        return cls(**data)


def generate_initial_population(size: int = 10, seed: int = 42) -> List[KernelGenome]:
    """Generate the initial population of kernel genomes.
    
    Args:
        size: Number of genomes in initial population
        seed: Random seed for deterministic generation
        
    Returns:
        List of KernelGenome instances
    """
    random.seed(seed)
    population = []
    current_time = time.time()
    
    # Define parameter ranges for diversity
    tile_sizes = [8, 16, 32, 64, 128]
    warp_counts = [1, 2, 4, 8, 16, 32]
    unroll_factors = [1, 2, 4, 8, 16]
    async_depths = [1, 2, 4, 8]
    precisions = ["fp32", "fp16", "bf16", "fp8", "int8"]
    
    for i in range(size):
        genome = KernelGenome(
            genome_id=f"gen0_kernel_{i:03d}",
            tile_size=random.choice(tile_sizes),
            warp_count=random.choice(warp_counts),
            unroll_factor=random.choice(unroll_factors),
            async_depth=random.choice(async_depths),
            precision=random.choice(precisions),
            timestamp=current_time,
            generation=0,
            fitness=0.0
        )
        population.append(genome)
    
    return population


def save_population(population: List[KernelGenome], output_dir: Path) -> None:
    """Save population to disk.
    
    Args:
        population: List of genomes to save
        output_dir: Directory to save genomes
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual genomes
    for genome in population:
        genome_path = output_dir / f"{genome.genome_id}.json"
        with open(genome_path, 'w') as f:
            json.dump(genome.to_dict(), f, indent=2)
    
    # Save population index
    index_path = output_dir / "population_index.json"
    index_data = {
        "size": len(population),
        "generation": 0,
        "timestamp": time.time(),
        "genomes": [g.genome_id for g in population]
    }
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_population(genomes_dir: Path) -> List[KernelGenome]:
    """Load population from disk.
    
    Args:
        genomes_dir: Directory containing genome files
        
    Returns:
        List of loaded KernelGenome instances
    """
    index_path = genomes_dir / "population_index.json"
    if not index_path.exists():
        return []
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    population = []
    for genome_id in index_data["genomes"]:
        genome_path = genomes_dir / f"{genome_id}.json"
        if genome_path.exists():
            with open(genome_path, 'r') as f:
                genome_data = json.load(f)
            population.append(KernelGenome.from_dict(genome_data))
    
    return population


def main() -> None:
    """Main entry point for initializing kernel population."""
    print("Initializing Phase III Autonomous Kernel Evolution")
    print("=" * 60)
    
    # Generate initial population
    population = generate_initial_population(size=10, seed=42)
    print(f"Generated {len(population)} initial kernel genomes")
    
    # Save to disk
    base_path = Path(__file__).parent / "genomes"
    save_population(population, base_path)
    print(f"Saved population to: {base_path}")
    
    # Display sample genomes
    print("\nSample Genomes:")
    for i, genome in enumerate(population[:3]):
        print(f"\n  Genome {i+1}: {genome.genome_id}")
        print(f"    Tile Size:      {genome.tile_size}")
        print(f"    Warp Count:     {genome.warp_count}")
        print(f"    Unroll Factor:  {genome.unroll_factor}")
        print(f"    Async Depth:    {genome.async_depth}")
        print(f"    Precision:      {genome.precision}")
    
    print("\n" + "=" * 60)
    print("Phase III initialization complete!")


if __name__ == "__main__":
    main()

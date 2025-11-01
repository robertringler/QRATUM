"""Quantum-inspired kernel search using Ising-like Hamiltonian encoding.

Encodes kernel configuration space as an Ising model and uses simulated
annealing to find low-energy (optimal) configurations.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class IsingConfiguration:
    """Represents a configuration in the Ising model."""
    
    config_id: str
    spins: List[int]  # +1 or -1 for each parameter
    energy: float = 0.0
    temperature: float = 1.0
    
    def to_kernel_params(self) -> Dict[str, int]:
        """Convert Ising spins to kernel parameters."""
        # Map spins to actual kernel parameters
        # Spin configuration encodes tile_size, warp_count, unroll, async_depth
        tile_size = 16 * (2 ** max(0, sum(self.spins[:3]) // 2))
        warp_count = max(1, 4 + sum(self.spins[3:6]))
        unroll_factor = max(1, 2 + sum(self.spins[6:9]) // 2)
        async_depth = max(1, 2 + sum(self.spins[9:12]) // 2)
        
        return {
            "tile_size": int(min(128, tile_size)),
            "warp_count": int(min(32, warp_count)),
            "unroll_factor": int(min(16, unroll_factor)),
            "async_depth": int(min(8, async_depth))
        }


@dataclass
class IsingModel:
    """Ising model for kernel configuration space."""
    
    num_spins: int
    coupling_matrix: List[List[float]]
    external_field: List[float]
    
    def compute_energy(self, configuration: IsingConfiguration) -> float:
        """Compute Hamiltonian energy for a configuration.
        
        H = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
        
        Args:
            configuration: Ising configuration
            
        Returns:
            Energy value (lower is better)
        """
        spins = configuration.spins
        energy = 0.0
        
        # Coupling term: -Σᵢⱼ Jᵢⱼ sᵢ sⱼ
        for i in range(self.num_spins):
            for j in range(i + 1, self.num_spins):
                energy -= self.coupling_matrix[i][j] * spins[i] * spins[j]
        
        # External field term: -Σᵢ hᵢ sᵢ
        for i in range(self.num_spins):
            energy -= self.external_field[i] * spins[i]
        
        return energy


class SimulatedAnnealing:
    """Simulated annealing optimizer for Ising model."""
    
    def __init__(self, model: IsingModel, initial_temperature: float = 10.0,
                 cooling_rate: float = 0.95):
        """Initialize simulated annealing optimizer.
        
        Args:
            model: Ising model to optimize
            initial_temperature: Starting temperature
            cooling_rate: Temperature decay rate per iteration
        """
        self.model = model
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.optimization_history: List[Dict] = []
        
    def generate_random_configuration(self) -> IsingConfiguration:
        """Generate a random spin configuration."""
        spins = [random.choice([-1, 1]) for _ in range(self.model.num_spins)]
        config = IsingConfiguration(
            config_id="random",
            spins=spins,
            temperature=self.initial_temperature
        )
        config.energy = self.model.compute_energy(config)
        return config
    
    def flip_random_spin(self, configuration: IsingConfiguration) -> IsingConfiguration:
        """Create a new configuration by flipping a random spin.
        
        Args:
            configuration: Current configuration
            
        Returns:
            New configuration with one flipped spin
        """
        new_spins = configuration.spins.copy()
        flip_index = random.randint(0, len(new_spins) - 1)
        new_spins[flip_index] *= -1
        
        new_config = IsingConfiguration(
            config_id=f"{configuration.config_id}_flip",
            spins=new_spins,
            temperature=configuration.temperature
        )
        new_config.energy = self.model.compute_energy(new_config)
        return new_config
    
    def accept_move(self, current_energy: float, new_energy: float, 
                   temperature: float) -> bool:
        """Decide whether to accept a move based on Metropolis criterion.
        
        Args:
            current_energy: Energy of current configuration
            new_energy: Energy of proposed configuration
            temperature: Current temperature
            
        Returns:
            True if move should be accepted
        """
        if new_energy < current_energy:
            return True
        
        # Accept with probability exp(-ΔE/T)
        delta_energy = new_energy - current_energy
        probability = math.exp(-delta_energy / temperature)
        return random.random() < probability
    
    def optimize(self, iterations: int = 1000, seed: int = 42) -> IsingConfiguration:
        """Run simulated annealing optimization.
        
        Args:
            iterations: Number of annealing iterations
            seed: Random seed for reproducibility
            
        Returns:
            Optimized configuration with lowest energy
        """
        random.seed(seed)
        
        # Initialize with random configuration
        current_config = self.generate_random_configuration()
        best_config = current_config
        
        temperature = self.initial_temperature
        
        for iteration in range(iterations):
            # Generate neighbor configuration
            neighbor_config = self.flip_random_spin(current_config)
            neighbor_config.temperature = temperature
            
            # Accept or reject move
            if self.accept_move(current_config.energy, neighbor_config.energy, temperature):
                current_config = neighbor_config
                
                # Update best if improved
                if current_config.energy < best_config.energy:
                    best_config = current_config
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                self.optimization_history.append({
                    "iteration": iteration + 1,
                    "temperature": temperature,
                    "current_energy": current_config.energy,
                    "best_energy": best_config.energy
                })
        
        best_config.config_id = "optimized"
        return best_config


class QuantumInspiredOptimizer:
    """High-level optimizer using quantum-inspired search."""
    
    def __init__(self, history_dir: Optional[Path] = None):
        """Initialize quantum-inspired optimizer.
        
        Args:
            history_dir: Directory to save optimization history
        """
        self.history_dir = history_dir or Path(__file__).parent
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
    def create_ising_model(self, num_params: int = 12) -> IsingModel:
        """Create Ising model for kernel configuration space.
        
        Args:
            num_params: Number of spin variables
            
        Returns:
            IsingModel instance
        """
        # Initialize coupling matrix (prefer similar parameters)
        coupling_matrix = [
            [random.uniform(-0.5, 0.5) if i != j else 0.0 
             for j in range(num_params)]
            for i in range(num_params)
        ]
        
        # External field biases certain configurations
        external_field = [random.uniform(-0.2, 0.2) for _ in range(num_params)]
        
        return IsingModel(
            num_spins=num_params,
            coupling_matrix=coupling_matrix,
            external_field=external_field
        )
    
    def optimize_kernel_config(self, kernel_id: str, iterations: int = 1000) -> Dict[str, int]:
        """Optimize kernel configuration using quantum-inspired search.
        
        Args:
            kernel_id: Kernel identifier
            iterations: Number of optimization iterations
            
        Returns:
            Optimized kernel parameters
        """
        # Create Ising model
        model = self.create_ising_model()
        
        # Run simulated annealing
        annealer = SimulatedAnnealing(model)
        optimized_config = annealer.optimize(iterations=iterations)
        
        # Convert to kernel parameters
        kernel_params = optimized_config.to_kernel_params()
        
        # Save optimization history
        history_path = self.history_dir / f"{kernel_id}_optimization.json"
        history_data = {
            "kernel_id": kernel_id,
            "iterations": iterations,
            "final_energy": optimized_config.energy,
            "kernel_params": kernel_params,
            "history": annealer.optimization_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        return kernel_params


if __name__ == "__main__":
    print("Quantum-Inspired Kernel Search Demo")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = QuantumInspiredOptimizer()
    
    # Run optimization
    kernel_id = "matmul_kernel_001"
    print(f"\nOptimizing {kernel_id}...")
    print("Running simulated annealing on Ising model...")
    
    optimized_params = optimizer.optimize_kernel_config(kernel_id, iterations=1000)
    
    print("\n" + "=" * 60)
    print("Optimized Kernel Parameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value}")
    
    history_path = optimizer.history_dir / f"{kernel_id}_optimization.json"
    print(f"\nOptimization history saved to: {history_path}")

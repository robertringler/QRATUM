"""Quantum-inspired Ising model optimization for kernel search."""
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
    
    spins: List[int]  # +1 or -1 for each parameter
    energy: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "spins": self.spins,
            "energy": self.energy,
        }


@dataclass
class OptimizationHistory:
    """History of optimization process."""
    
    iterations: int = 0
    configurations: List[IsingConfiguration] = field(default_factory=list)
    best_energy: float = float("inf")
    best_configuration: IsingConfiguration | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "iterations": self.iterations,
            "configurations": [c.to_dict() for c in self.configurations[-100:]],  # Last 100
            "best_energy": self.best_energy,
            "best_configuration": (
                self.best_configuration.to_dict() if self.best_configuration else None
            ),
        }


class IsingOptimizer:
    """
    Quantum-inspired optimizer using Ising model formulation.
    Uses simulated annealing to find minimum energy configuration.
    """
    
    def __init__(self, num_parameters: int = 5, seed: int = 42):
        self.num_parameters = num_parameters
        self.history = OptimizationHistory()
        random.seed(seed)
        
        # Initialize coupling matrix (interactions between parameters)
        self.couplings = self._init_couplings()
        
    def _init_couplings(self) -> List[List[float]]:
        """Initialize random coupling matrix."""
        couplings = []
        for i in range(self.num_parameters):
            row = []
            for j in range(self.num_parameters):
                if i == j:
                    row.append(0.0)
                else:
                    # Random coupling strength between -1 and 1
                    row.append(random.uniform(-1.0, 1.0))
            couplings.append(row)
        return couplings
        
    def compute_energy(self, config: IsingConfiguration) -> float:
        """
        Compute Hamiltonian energy for a configuration.
        E = -Σ(i,j) J_ij * s_i * s_j - Σ_i h_i * s_i
        """
        energy = 0.0
        
        # Interaction term
        for i in range(self.num_parameters):
            for j in range(i + 1, self.num_parameters):
                energy -= self.couplings[i][j] * config.spins[i] * config.spins[j]
                
        # External field term (bias toward certain configurations)
        for i in range(self.num_parameters):
            # Bias parameters toward +1 (higher values)
            energy -= 0.1 * config.spins[i]
            
        return energy
        
    def simulated_annealing(
        self,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        steps_per_temp: int = 100,
    ) -> IsingConfiguration:
        """
        Perform simulated annealing to find minimum energy configuration.
        """
        # Initialize random configuration
        current_config = IsingConfiguration(
            spins=[random.choice([-1, 1]) for _ in range(self.num_parameters)]
        )
        current_config.energy = self.compute_energy(current_config)
        
        best_config = IsingConfiguration(
            spins=current_config.spins.copy(),
            energy=current_config.energy,
        )
        
        temperature = initial_temp
        total_iterations = 0
        
        while temperature > final_temp:
            for _ in range(steps_per_temp):
                # Propose a spin flip
                flip_index = random.randint(0, self.num_parameters - 1)
                new_spins = current_config.spins.copy()
                new_spins[flip_index] *= -1
                
                new_config = IsingConfiguration(spins=new_spins)
                new_config.energy = self.compute_energy(new_config)
                
                # Accept or reject based on Metropolis criterion
                delta_energy = new_config.energy - current_config.energy
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                    current_config = new_config
                    
                    # Update best if better
                    if current_config.energy < best_config.energy:
                        best_config = IsingConfiguration(
                            spins=current_config.spins.copy(),
                            energy=current_config.energy,
                        )
                        
                total_iterations += 1
                
                # Record configuration periodically
                if total_iterations % 50 == 0:
                    self.history.configurations.append(
                        IsingConfiguration(
                            spins=current_config.spins.copy(),
                            energy=current_config.energy,
                        )
                    )
                    
            # Cool down
            temperature *= cooling_rate
            
        self.history.iterations = total_iterations
        self.history.best_energy = best_config.energy
        self.history.best_configuration = best_config
        
        return best_config
        
    def spins_to_kernel_config(self, spins: List[int]) -> Dict[str, int]:
        """
        Convert Ising spins to kernel configuration parameters.
        Maps binary decisions to discrete parameter values.
        """
        # Map spins to parameter ranges
        # Spin +1 -> high value, Spin -1 -> low value
        
        tile_size = 512 if spins[0] > 0 else 256
        warp_count = 32 if spins[1] > 0 else 16
        unroll_factor = 8 if spins[2] > 0 else 4
        async_depth = 4 if spins[3] > 0 else 2
        
        # Additional parameter if available
        prefetch = 4 if len(spins) > 4 and spins[4] > 0 else 2
        
        return {
            "tile_size": tile_size,
            "warp_count": warp_count,
            "unroll_factor": unroll_factor,
            "async_depth": async_depth,
            "prefetch_distance": prefetch,
        }
        
    def optimize_kernel_config(self) -> Tuple[Dict[str, int], float]:
        """
        Run quantum-inspired optimization to find best kernel configuration.
        Returns (config_dict, energy).
        """
        best_config = self.simulated_annealing()
        kernel_config = self.spins_to_kernel_config(best_config.spins)
        
        return kernel_config, best_config.energy
        
    def save_history(self, output_path: str = "quantum_search/history.json") -> Path:
        """Save optimization history to disk."""
        history_path = Path(output_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)
            
        return history_path

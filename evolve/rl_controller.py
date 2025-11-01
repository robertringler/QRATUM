"""Reinforcement learning controller for autonomous kernel optimization.

This module implements a simplified RL agent (inspired by PPO/DDPG) that learns
to adjust kernel parameters based on performance feedback.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve.init_population import KernelGenome
from evolve.introspection_agent import PerformanceMetrics


@dataclass
class RLState:
    """State representation for RL agent."""
    
    tile_size: int
    warp_count: int
    unroll_factor: int
    async_depth: int
    avg_latency_ms: float
    cache_miss_rate: float
    warp_divergence_pct: float
    
    def to_vector(self) -> List[float]:
        """Convert state to normalized feature vector."""
        return [
            self.tile_size / 128.0,
            self.warp_count / 32.0,
            self.unroll_factor / 16.0,
            self.async_depth / 8.0,
            min(self.avg_latency_ms / 100.0, 1.0),
            self.cache_miss_rate,
            self.warp_divergence_pct / 100.0
        ]


@dataclass
class RLAction:
    """Action representation for RL agent."""
    
    delta_tile_size: int = 0
    delta_warp_count: int = 0
    delta_unroll_factor: int = 0
    delta_async_depth: int = 0


class RLPolicy:
    """Simple policy network for kernel parameter optimization."""
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95):
        """Initialize the RL policy.
        
        Args:
            learning_rate: Learning rate for policy updates
            discount_factor: Discount factor for future rewards
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Simple linear policy weights (state_dim=7, action_dim=4)
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(7)] for _ in range(4)]
        self.training_history: List[Dict] = []
        
    def select_action(self, state: RLState, epsilon: float = 0.1) -> RLAction:
        """Select an action based on current policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate for epsilon-greedy policy
            
        Returns:
            RLAction to take
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return RLAction(
                delta_tile_size=random.randint(-2, 2),
                delta_warp_count=random.randint(-1, 1),
                delta_unroll_factor=random.randint(-1, 1),
                delta_async_depth=random.randint(-1, 1)
            )
        
        # Exploit: use policy network
        state_vec = state.to_vector()
        action_values = [sum(w * s for w, s in zip(weights_row, state_vec)) 
                        for weights_row in self.weights]
        
        # Convert continuous outputs to discrete actions
        return RLAction(
            delta_tile_size=int(action_values[0] * 2),
            delta_warp_count=int(action_values[1]),
            delta_unroll_factor=int(action_values[2]),
            delta_async_depth=int(action_values[3])
        )
    
    def compute_reward(self, metrics: PerformanceMetrics, baseline_latency: float = 10.0) -> float:
        """Compute reward signal from performance metrics.
        
        Args:
            metrics: Performance metrics from kernel execution
            baseline_latency: Baseline latency for normalization
            
        Returns:
            Reward value (higher is better)
        """
        # Reward inversely proportional to latency, penalize divergence and cache misses
        latency_reward = baseline_latency / max(metrics.latency_ms, 0.1)
        divergence_penalty = -0.1 * metrics.warp_divergence_pct
        cache_penalty = -1.0 * metrics.cache_miss_rate
        
        return latency_reward + divergence_penalty + cache_penalty
    
    def update_policy(self, state: RLState, action: RLAction, reward: float) -> None:
        """Update policy weights based on observed reward.
        
        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
        """
        state_vec = state.to_vector()
        action_vec = [action.delta_tile_size, action.delta_warp_count, 
                     action.delta_unroll_factor, action.delta_async_depth]
        
        # Simple gradient update (policy gradient approximation)
        for i, delta in enumerate(action_vec):
            if delta != 0:
                gradient = reward * delta
                for j, s in enumerate(state_vec):
                    self.weights[i][j] += self.learning_rate * gradient * s
        
        # Log training step
        self.training_history.append({
            "state": state.to_vector(),
            "action": asdict(action),
            "reward": reward
        })
    
    def save(self, path: Path) -> None:
        """Save policy to disk.
        
        Args:
            path: Path to save policy
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        policy_data = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "weights": self.weights,
            "training_steps": len(self.training_history)
        }
        with open(path, 'w') as f:
            json.dump(policy_data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> RLPolicy:
        """Load policy from disk.
        
        Args:
            path: Path to load policy from
            
        Returns:
            Loaded RLPolicy instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        policy = cls(
            learning_rate=data["learning_rate"],
            discount_factor=data["discount_factor"]
        )
        policy.weights = data["weights"]
        return policy


class RLController:
    """High-level controller for RL-based kernel optimization."""
    
    def __init__(self, policy: Optional[RLPolicy] = None):
        """Initialize the RL controller.
        
        Args:
            policy: Optional pre-trained policy
        """
        self.policy = policy or RLPolicy()
        self.optimization_history: List[Dict] = []
        
    def optimize_kernel(self, genome: KernelGenome, metrics: PerformanceMetrics,
                       baseline_latency: float = 10.0) -> KernelGenome:
        """Optimize kernel parameters based on performance metrics.
        
        Args:
            genome: Current kernel genome
            metrics: Performance metrics from execution
            baseline_latency: Baseline latency for reward computation
            
        Returns:
            Optimized KernelGenome
        """
        # Create state from genome and metrics
        state = RLState(
            tile_size=genome.tile_size,
            warp_count=genome.warp_count,
            unroll_factor=genome.unroll_factor,
            async_depth=genome.async_depth,
            avg_latency_ms=metrics.latency_ms,
            cache_miss_rate=metrics.cache_miss_rate,
            warp_divergence_pct=metrics.warp_divergence_pct
        )
        
        # Select action
        action = self.policy.select_action(state, epsilon=0.15)
        
        # Compute reward
        reward = self.policy.compute_reward(metrics, baseline_latency)
        
        # Update policy
        self.policy.update_policy(state, action, reward)
        
        # Apply action to create new genome
        new_genome = KernelGenome(
            genome_id=f"{genome.genome_id}_rl",
            tile_size=max(8, min(128, genome.tile_size + action.delta_tile_size * 8)),
            warp_count=max(1, min(32, genome.warp_count + action.delta_warp_count)),
            unroll_factor=max(1, min(16, genome.unroll_factor + action.delta_unroll_factor)),
            async_depth=max(1, min(8, genome.async_depth + action.delta_async_depth)),
            precision=genome.precision,
            timestamp=metrics.timestamp,
            generation=genome.generation + 1,
            fitness=reward
        )
        
        # Log optimization step
        self.optimization_history.append({
            "genome_id": genome.genome_id,
            "new_genome_id": new_genome.genome_id,
            "reward": reward,
            "action": asdict(action)
        })
        
        return new_genome
    
    def train_loop(self, population: List[KernelGenome], 
                  metrics_list: List[PerformanceMetrics],
                  epochs: int = 10) -> List[KernelGenome]:
        """Run training loop on population.
        
        Args:
            population: List of kernel genomes
            metrics_list: Corresponding performance metrics
            epochs: Number of training epochs
            
        Returns:
            Optimized population
        """
        if len(population) != len(metrics_list):
            raise ValueError("Population and metrics must have same length")
        
        current_population = population
        
        for epoch in range(epochs):
            new_population = []
            for genome, metrics in zip(current_population, metrics_list):
                optimized = self.optimize_kernel(genome, metrics)
                new_population.append(optimized)
            current_population = new_population
        
        return current_population
    
    def save_controller(self, path: Path) -> None:
        """Save controller state.
        
        Args:
            path: Path to save controller
        """
        policy_path = path / "policy.json"
        self.policy.save(policy_path)
        
        history_path = path / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)


if __name__ == "__main__":
    from evolve.init_population import generate_initial_population
    from evolve.introspection_agent import simulate_kernel_execution, IntrospectionAgent
    
    print("RL Controller Demo")
    print("=" * 60)
    
    # Generate test population
    population = generate_initial_population(size=5, seed=42)
    
    # Create introspection agent
    agent = IntrospectionAgent()
    
    # Simulate executions and collect metrics
    metrics_list = []
    for genome in population:
        metrics = simulate_kernel_execution(agent, genome.genome_id)
        metrics_list.append(metrics)
    
    # Initialize controller
    controller = RLController()
    
    # Run optimization
    print("\nRunning RL optimization...")
    optimized_population = controller.train_loop(population, metrics_list, epochs=3)
    
    print("\nOptimization Results:")
    for orig, opt in zip(population, optimized_population):
        print(f"\nOriginal: {orig.genome_id}")
        print(f"  Tile: {orig.tile_size} -> {opt.tile_size}")
        print(f"  Warp: {orig.warp_count} -> {opt.warp_count}")
        print(f"  Fitness: {orig.fitness:.3f} -> {opt.fitness:.3f}")
    
    # Save controller
    save_path = Path(__file__).parent / "policies"
    controller.save_controller(save_path)
    print(f"\nController saved to: {save_path}")

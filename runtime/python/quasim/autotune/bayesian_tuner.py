"""Bayesian optimizer for kernel configuration tuning."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TuningConfig:
    """Configuration space for tuning."""
    name: str
    param_ranges: Dict[str, Tuple[float, float]]
    objectives: List[str] = field(default_factory=lambda: ["latency", "energy"])
    max_iterations: int = 100
    exploration_weight: float = 2.0


@dataclass
class SamplePoint:
    """A sampled configuration with its observed metrics."""
    config: Dict[str, float]
    metrics: Dict[str, float]
    iteration: int


class BayesianTuner:
    """Bayesian optimizer for multi-objective kernel tuning."""
    
    def __init__(self, tuning_config: TuningConfig) -> None:
        self.config = tuning_config
        self._samples: List[SamplePoint] = []
        self._best_config: Optional[Dict[str, float]] = None
        self._best_score: float = float("inf")
        
    def _sample_random_config(self) -> Dict[str, float]:
        """Sample a random configuration from the parameter space."""
        config = {}
        for param, (min_val, max_val) in self.config.param_ranges.items():
            # Log-uniform sampling for better exploration
            log_min = math.log(max(min_val, 1e-10))
            log_max = math.log(max_val)
            config[param] = math.exp(random.uniform(log_min, log_max))
        return config
        
    def _acquisition_function(self, config: Dict[str, float]) -> float:
        """Upper confidence bound acquisition function."""
        if not self._samples:
            return 0.0
            
        # Compute mean and variance from samples (simplified GP approximation)
        similarities = []
        scores = []
        
        for sample in self._samples:
            # Compute distance to sample
            dist = sum(
                (config.get(k, 0) - sample.config.get(k, 0)) ** 2
                for k in self.config.param_ranges
            )
            similarity = math.exp(-dist / 10.0)
            similarities.append(similarity)
            
            # Compute combined score
            score = sum(sample.metrics.get(obj, 0) for obj in self.config.objectives)
            scores.append(score)
            
        # Weighted mean and variance
        total_sim = sum(similarities) + 1e-10
        mean = sum(s * w for s, w in zip(scores, similarities)) / total_sim
        variance = sum(
            ((score - mean) ** 2) * sim
            for score, sim in zip(scores, similarities)
        ) / total_sim
        std = math.sqrt(variance + 1e-10)
        
        # UCB = mean - exploration_weight * std (minimize)
        return mean - self.config.exploration_weight * std
        
    def _propose_config(self) -> Dict[str, float]:
        """Propose next configuration to evaluate."""
        # Random exploration for first few iterations
        if len(self._samples) < 10:
            return self._sample_random_config()
            
        # Generate candidates and select best according to acquisition
        candidates = [self._sample_random_config() for _ in range(20)]
        best_candidate = min(candidates, key=self._acquisition_function)
        return best_candidate
        
    def tune(
        self,
        objective_func: Callable[[Dict[str, float]], Dict[str, float]],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run Bayesian optimization to find optimal configuration."""
        for iteration in range(self.config.max_iterations):
            # Propose configuration
            config = self._propose_config()
            
            # Evaluate objective
            metrics = objective_func(config)
            
            # Record sample
            sample = SamplePoint(
                config=config,
                metrics=metrics,
                iteration=iteration,
            )
            self._samples.append(sample)
            
            # Update best configuration
            score = sum(metrics.get(obj, 0) for obj in self.config.objectives)
            if score < self._best_score:
                self._best_score = score
                self._best_config = config.copy()
                
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: best_score={self._best_score:.4f}")
                
        return self._best_config or self._sample_random_config()
        
    def get_optimization_history(self) -> List[SamplePoint]:
        """Get history of sampled configurations."""
        return self._samples.copy()
        
    def get_pareto_frontier(self) -> List[SamplePoint]:
        """Get Pareto-optimal configurations for multi-objective tuning."""
        if len(self.config.objectives) < 2:
            return [min(self._samples, key=lambda s: s.metrics.get(self.config.objectives[0], float("inf")))]
            
        pareto_frontier = []
        
        for sample in self._samples:
            dominated = False
            for other in self._samples:
                if sample == other:
                    continue
                    
                # Check if other dominates sample
                better_in_all = all(
                    other.metrics.get(obj, float("inf")) <= sample.metrics.get(obj, float("inf"))
                    for obj in self.config.objectives
                )
                strictly_better = any(
                    other.metrics.get(obj, float("inf")) < sample.metrics.get(obj, float("inf"))
                    for obj in self.config.objectives
                )
                
                if better_in_all and strictly_better:
                    dominated = True
                    break
                    
            if not dominated:
                pareto_frontier.append(sample)
                
        return pareto_frontier

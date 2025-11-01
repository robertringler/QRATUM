"""Causal profiling and counterfactual benchmarking.

Implements perturbation profiling by injecting micro-delays and measuring
downstream latency shifts to estimate causal contribution of each function.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CausalInfluence:
    """Represents causal influence of a function on total runtime."""
    
    function_name: str
    baseline_latency_ms: float
    perturbed_latency_ms: float
    delay_injected_ms: float
    causal_impact: float = 0.0  # Normalized impact score
    
    def __post_init__(self):
        """Calculate causal impact after initialization."""
        if self.delay_injected_ms > 0:
            # Impact = (observed_change - injected_delay) / injected_delay
            observed_change = self.perturbed_latency_ms - self.baseline_latency_ms
            excess_change = observed_change - self.delay_injected_ms
            self.causal_impact = excess_change / self.delay_injected_ms
        else:
            self.causal_impact = 0.0


@dataclass
class CausalProfile:
    """Complete causal profile for a kernel execution."""
    
    kernel_id: str
    influences: List[CausalInfluence]
    total_baseline_ms: float
    timestamp: float = field(default_factory=time.time)
    
    def get_critical_path(self) -> List[str]:
        """Get functions on the critical path (highest causal impact).
        
        Returns:
            List of function names sorted by causal impact
        """
        sorted_influences = sorted(self.influences, 
                                  key=lambda x: abs(x.causal_impact), 
                                  reverse=True)
        return [inf.function_name for inf in sorted_influences]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kernel_id": self.kernel_id,
            "total_baseline_ms": self.total_baseline_ms,
            "timestamp": self.timestamp,
            "influences": [
                {
                    "function_name": inf.function_name,
                    "baseline_latency_ms": inf.baseline_latency_ms,
                    "perturbed_latency_ms": inf.perturbed_latency_ms,
                    "delay_injected_ms": inf.delay_injected_ms,
                    "causal_impact": inf.causal_impact
                }
                for inf in self.influences
            ]
        }


class CausalProfiler:
    """Profiler for measuring causal contributions using perturbation analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize causal profiler.
        
        Args:
            output_dir: Directory to save causal influence maps
        """
        self.output_dir = output_dir or Path(__file__).parent / "causal"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def measure_baseline(self, kernel_id: str, functions: List[str]) -> Dict[str, float]:
        """Measure baseline latency for each function.
        
        Args:
            kernel_id: Kernel identifier
            functions: List of function names to profile
            
        Returns:
            Dictionary mapping function names to baseline latencies
        """
        baselines = {}
        for func in functions:
            # Simulate function execution
            start = time.perf_counter()
            time.sleep(random.uniform(0.001, 0.005))  # Simulate work
            end = time.perf_counter()
            baselines[func] = (end - start) * 1000.0  # Convert to ms
        return baselines
    
    def inject_perturbation(self, kernel_id: str, target_function: str,
                           functions: List[str], delay_ms: float) -> Dict[str, float]:
        """Inject micro-delay into target function and measure downstream effects.
        
        Args:
            kernel_id: Kernel identifier
            target_function: Function to perturb
            functions: All functions in the execution
            delay_ms: Delay to inject in milliseconds
            
        Returns:
            Dictionary mapping function names to perturbed latencies
        """
        perturbed = {}
        for func in functions:
            start = time.perf_counter()
            
            # Inject delay if this is the target function
            if func == target_function:
                time.sleep(delay_ms / 1000.0)
            
            # Simulate function execution
            time.sleep(random.uniform(0.001, 0.005))
            
            end = time.perf_counter()
            perturbed[func] = (end - start) * 1000.0
        
        return perturbed
    
    def profile_kernel(self, kernel_id: str, functions: Optional[List[str]] = None,
                      perturbation_ms: float = 0.5) -> CausalProfile:
        """Perform causal profiling on a kernel.
        
        Args:
            kernel_id: Kernel identifier
            functions: List of functions to profile (generates default if None)
            perturbation_ms: Delay to inject for perturbation
            
        Returns:
            CausalProfile with influence measurements
        """
        if functions is None:
            functions = [
                f"{kernel_id}_init",
                f"{kernel_id}_compute",
                f"{kernel_id}_memory_transfer",
                f"{kernel_id}_sync",
                f"{kernel_id}_cleanup"
            ]
        
        # Measure baseline
        baselines = self.measure_baseline(kernel_id, functions)
        total_baseline = sum(baselines.values())
        
        # Profile each function with perturbation
        influences = []
        for target_func in functions:
            perturbed = self.inject_perturbation(kernel_id, target_func, 
                                                 functions, perturbation_ms)
            
            influence = CausalInfluence(
                function_name=target_func,
                baseline_latency_ms=baselines[target_func],
                perturbed_latency_ms=perturbed[target_func],
                delay_injected_ms=perturbation_ms
            )
            influences.append(influence)
        
        return CausalProfile(
            kernel_id=kernel_id,
            influences=influences,
            total_baseline_ms=total_baseline
        )
    
    def save_profile(self, profile: CausalProfile) -> Path:
        """Save causal profile to disk.
        
        Args:
            profile: CausalProfile to save
            
        Returns:
            Path to saved profile
        """
        profile_path = self.output_dir / f"{profile.kernel_id}_causal.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        return profile_path
    
    def load_profile(self, kernel_id: str) -> Optional[CausalProfile]:
        """Load causal profile from disk.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Loaded CausalProfile or None if not found
        """
        profile_path = self.output_dir / f"{kernel_id}_causal.json"
        if not profile_path.exists():
            return None
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        influences = [
            CausalInfluence(
                function_name=inf["function_name"],
                baseline_latency_ms=inf["baseline_latency_ms"],
                perturbed_latency_ms=inf["perturbed_latency_ms"],
                delay_injected_ms=inf["delay_injected_ms"]
            )
            for inf in data["influences"]
        ]
        
        return CausalProfile(
            kernel_id=data["kernel_id"],
            influences=influences,
            total_baseline_ms=data["total_baseline_ms"],
            timestamp=data["timestamp"]
        )
    
    def generate_influence_map(self, profile: CausalProfile) -> str:
        """Generate a text-based influence map.
        
        Args:
            profile: CausalProfile to visualize
            
        Returns:
            Formatted influence map
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"Causal Influence Map: {profile.kernel_id}")
        lines.append("=" * 70)
        lines.append(f"Total Baseline: {profile.total_baseline_ms:.3f} ms")
        lines.append("")
        lines.append("Critical Path (sorted by causal impact):")
        
        critical_path = profile.get_critical_path()
        for i, func_name in enumerate(critical_path[:5], 1):
            influence = next(inf for inf in profile.influences 
                           if inf.function_name == func_name)
            lines.append(f"\n{i}. {func_name}")
            lines.append(f"   Baseline:      {influence.baseline_latency_ms:.3f} ms")
            lines.append(f"   Perturbed:     {influence.perturbed_latency_ms:.3f} ms")
            lines.append(f"   Impact Score:  {influence.causal_impact:.3f}")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("Causal Profiler Demo")
    print("=" * 70)
    
    # Initialize profiler
    profiler = CausalProfiler()
    
    # Profile a kernel
    kernel_id = "matmul_kernel_001"
    print(f"\nProfiling {kernel_id}...")
    profile = profiler.profile_kernel(kernel_id, perturbation_ms=0.5)
    
    # Display influence map
    print("\n" + profiler.generate_influence_map(profile))
    
    # Save profile
    profile_path = profiler.save_profile(profile)
    print(f"\nCausal profile saved to: {profile_path}")

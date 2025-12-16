"""
Performance Profiler for QRATUM

Validates that optimizations preserve equivalence while improving performance.
Certificate: QRATUM-HARDENING-20251215-V5
"""

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..validation.equivalence import EquivalenceValidator


class PerformanceProfiler:
    """
    Profiles performance while ensuring equivalence preservation.
    
    Features:
    - Baseline vs optimized comparison
    - Mandatory equivalence checking
    - Detailed timing reports
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize performance profiler.
        
        Args:
            tolerance: Equivalence tolerance for validation
        """
        self.validator = EquivalenceValidator(tolerance=tolerance)
        self.profiles = []
    
    def profile_function(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        inputs: Dict[str, Any],
        name: str = "function",
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Profile and compare baseline vs optimized function.
        
        Args:
            baseline_fn: Baseline function
            optimized_fn: Optimized function
            inputs: Input arguments (dict for **kwargs)
            name: Name for reporting
            num_runs: Number of runs for timing average
            
        Returns:
            Dictionary with profiling results
        """
        # Run baseline
        baseline_times = []
        baseline_result = None
        for i in range(num_runs):
            start = time.perf_counter()
            result = baseline_fn(**inputs)
            end = time.perf_counter()
            baseline_times.append(end - start)
            if i == 0:
                baseline_result = result
        
        # Run optimized
        optimized_times = []
        optimized_result = None
        for i in range(num_runs):
            start = time.perf_counter()
            result = optimized_fn(**inputs)
            end = time.perf_counter()
            optimized_times.append(end - start)
            if i == 0:
                optimized_result = result
        
        # Compute timing statistics
        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)
        optimized_mean = np.mean(optimized_times)
        optimized_std = np.std(optimized_times)
        speedup = baseline_mean / optimized_mean if optimized_mean > 0 else 0
        
        # Validate equivalence
        if isinstance(baseline_result, np.ndarray) and isinstance(optimized_result, np.ndarray):
            equivalence = self.validator.validate_array_equivalence(
                baseline_result, optimized_result, "baseline", "optimized"
            )
        elif isinstance(baseline_result, dict) and isinstance(optimized_result, dict):
            equivalence = self.validator.validate_dict_equivalence(
                baseline_result, optimized_result
            )
        elif isinstance(baseline_result, (int, float)) and isinstance(optimized_result, (int, float)):
            equivalence = self.validator.validate_scalar_equivalence(
                baseline_result, optimized_result, "baseline", "optimized"
            )
        else:
            equivalence = {
                "equivalent": baseline_result == optimized_result,
                "reason": "Direct equality comparison"
            }
        
        profile_result = {
            "name": name,
            "num_runs": num_runs,
            "baseline": {
                "mean_time": float(baseline_mean),
                "std_time": float(baseline_std),
                "times": [float(t) for t in baseline_times]
            },
            "optimized": {
                "mean_time": float(optimized_mean),
                "std_time": float(optimized_std),
                "times": [float(t) for t in optimized_times]
            },
            "speedup": float(speedup),
            "equivalence": equivalence
        }
        
        self.profiles.append(profile_result)
        return profile_result
    
    def profile_single_function(
        self,
        fn: Callable,
        inputs: Dict[str, Any],
        name: str = "function",
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Profile a single function without comparison.
        
        Args:
            fn: Function to profile
            inputs: Input arguments
            name: Name for reporting
            num_runs: Number of runs
            
        Returns:
            Dictionary with profiling results
        """
        times = []
        result = None
        for i in range(num_runs):
            start = time.perf_counter()
            result = fn(**inputs)
            end = time.perf_counter()
            times.append(end - start)
            if i == 0:
                result_first = result
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        profile_result = {
            "name": name,
            "num_runs": num_runs,
            "mean_time": float(mean_time),
            "std_time": float(std_time),
            "times": [float(t) for t in times]
        }
        
        self.profiles.append(profile_result)
        return profile_result
    
    def get_profiles(self) -> list:
        """Get all profiling results."""
        return self.profiles.copy()
    
    def reset_profiles(self) -> None:
        """Clear all profiling results."""
        self.profiles.clear()
    
    def generate_report(self) -> str:
        """
        Generate a formatted profiling report.
        
        Returns:
            Formatted report string
        """
        if not self.profiles:
            return "No profiling data available."
        
        lines = ["Performance Profiling Report", "=" * 50, ""]
        
        for profile in self.profiles:
            lines.append(f"Function: {profile['name']}")
            lines.append(f"  Runs: {profile['num_runs']}")
            
            if "baseline" in profile:
                lines.append(f"  Baseline: {profile['baseline']['mean_time']:.6f}s ± {profile['baseline']['std_time']:.6f}s")
                lines.append(f"  Optimized: {profile['optimized']['mean_time']:.6f}s ± {profile['optimized']['std_time']:.6f}s")
                lines.append(f"  Speedup: {profile['speedup']:.2f}x")
                lines.append(f"  Equivalent: {profile['equivalence']['equivalent']}")
            else:
                lines.append(f"  Time: {profile['mean_time']:.6f}s ± {profile['std_time']:.6f}s")
            
            lines.append("")
        
        return "\n".join(lines)

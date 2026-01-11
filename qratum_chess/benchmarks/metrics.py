"""Performance metrics for QRATUM-Chess benchmarking.

Defines core KPIs and measurement infrastructure:
- Nodes/sec (single core and multi-threaded)
- MCTS rollouts/sec
- NN eval latency
- Hash table hit rate
- Branch misprediction rate
- Cache miss rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class PerformanceTargets:
    """Performance targets for QRATUM-Chess certification.
    
    Based on Stage IV specifications.
    """
    # Core KPIs
    nodes_per_sec_single: int = 70_000_000  # ≥ 70M nodes/sec single core
    nodes_per_sec_32thread: int = 1_900_000_000  # ≥ 1.9B nodes/sec 32 threads
    mcts_rollouts_per_sec: int = 500_000  # ≥ 500k rollouts/sec
    nn_eval_latency_ms: float = 0.15  # ≤ 0.15 ms
    hash_hit_rate: float = 0.93  # ≥ 93%
    branch_misprediction_rate: float = 0.04  # ≤ 4%
    cache_l3_miss_rate: float = 0.08  # ≤ 8%


@dataclass
class PerformanceResult:
    """Result from a single performance measurement."""
    metric_name: str
    value: float
    unit: str
    target: float | None = None
    passed: bool | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, target: float, higher_is_better: bool = True) -> None:
        """Evaluate if this result meets the target."""
        self.target = target
        if higher_is_better:
            self.passed = self.value >= target
        else:
            self.passed = self.value <= target


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    test_name: str
    results: list[PerformanceResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    overall_passed: bool = True
    
    def add_result(self, result: PerformanceResult) -> None:
        """Add a result and update overall status."""
        self.results.append(result)
        if result.passed is False:
            self.overall_passed = False
    
    def finalize(self) -> None:
        """Finalize the report."""
        self.end_time = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_name": self.test_name,
            "overall_passed": self.overall_passed,
            "duration_seconds": (self.end_time or time.time()) - self.start_time,
            "results": [
                {
                    "metric": r.metric_name,
                    "value": r.value,
                    "unit": r.unit,
                    "target": r.target,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


class PerformanceMetrics:
    """Performance measurement infrastructure.
    
    Provides methods for measuring various performance metrics
    of the chess engine.
    """
    
    def __init__(self, targets: PerformanceTargets | None = None):
        """Initialize metrics collector.
        
        Args:
            targets: Performance targets for evaluation.
        """
        self.targets = targets or PerformanceTargets()
    
    def measure_nodes_per_second(
        self,
        engine,
        position,
        duration_seconds: float = 1.0,
        num_threads: int = 1
    ) -> PerformanceResult:
        """Measure nodes searched per second.
        
        Args:
            engine: Chess engine instance with search capability.
            position: Position to search from.
            duration_seconds: Duration of measurement.
            num_threads: Number of threads to use.
            
        Returns:
            Performance result with nodes/sec measurement.
        """
        time_limit_ms = duration_seconds * 1000
        
        start = time.perf_counter()
        _, _, stats = engine.search(position, time_limit_ms=time_limit_ms)
        elapsed = time.perf_counter() - start
        
        total_nodes = stats.nodes_searched + getattr(stats, 'quiescence_nodes', 0)
        nps = total_nodes / elapsed if elapsed > 0 else 0
        
        target = (
            self.targets.nodes_per_sec_single if num_threads == 1
            else self.targets.nodes_per_sec_32thread
        )
        
        result = PerformanceResult(
            metric_name=f"nodes_per_sec_{num_threads}t",
            value=nps,
            unit="nodes/sec",
            metadata={"threads": num_threads, "duration": elapsed},
        )
        result.evaluate(target, higher_is_better=True)
        
        return result
    
    def measure_mcts_rollouts(
        self,
        mcts_engine,
        position,
        duration_seconds: float = 1.0
    ) -> PerformanceResult:
        """Measure MCTS rollouts per second.
        
        Args:
            mcts_engine: MCTS engine instance.
            position: Position to search from.
            duration_seconds: Duration of measurement.
            
        Returns:
            Performance result with rollouts/sec measurement.
        """
        time_limit_ms = duration_seconds * 1000
        
        start = time.perf_counter()
        _, visit_counts, _ = mcts_engine.search(position, time_limit_ms=time_limit_ms)
        elapsed = time.perf_counter() - start
        
        total_visits = sum(visit_counts.values()) if visit_counts else 0
        rollouts_per_sec = total_visits / elapsed if elapsed > 0 else 0
        
        result = PerformanceResult(
            metric_name="mcts_rollouts_per_sec",
            value=rollouts_per_sec,
            unit="rollouts/sec",
            metadata={"duration": elapsed, "total_visits": total_visits},
        )
        result.evaluate(self.targets.mcts_rollouts_per_sec, higher_is_better=True)
        
        return result
    
    def measure_nn_eval_latency(
        self,
        evaluator,
        positions: list,
        num_samples: int = 1000
    ) -> PerformanceResult:
        """Measure neural network evaluation latency.
        
        Args:
            evaluator: Neural evaluator instance.
            positions: List of positions to evaluate.
            num_samples: Number of evaluation samples.
            
        Returns:
            Performance result with latency measurement.
        """
        if not positions or evaluator is None:
            return PerformanceResult(
                metric_name="nn_eval_latency",
                value=0.0,
                unit="ms",
                metadata={"skipped": True, "reason": "No evaluator or positions"},
            )
        
        latencies = []
        for i in range(num_samples):
            pos = positions[i % len(positions)]
            start = time.perf_counter()
            evaluator.evaluate(pos)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        
        result = PerformanceResult(
            metric_name="nn_eval_latency",
            value=avg_latency,
            unit="ms",
            metadata={
                "samples": num_samples,
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
            },
        )
        result.evaluate(self.targets.nn_eval_latency_ms, higher_is_better=False)
        
        return result
    
    def measure_hash_hit_rate(
        self,
        engine,
        positions: list,
        depth: int = 8
    ) -> PerformanceResult:
        """Measure transposition table hit rate.
        
        Args:
            engine: Search engine with transposition table.
            positions: Positions to search.
            depth: Search depth.
            
        Returns:
            Performance result with hit rate measurement.
        """
        total_lookups = 0
        total_hits = 0
        
        for pos in positions[:100]:  # Limit for efficiency
            # Clear tables if available
            if hasattr(engine, 'clear_tables'):
                engine.clear_tables()
            _, _, stats = engine.search(pos, depth=depth)
            total_hits += getattr(stats, 'tt_hits', 0)
            total_lookups += getattr(stats, 'nodes_searched', 0)
        
        hit_rate = total_hits / total_lookups if total_lookups > 0 else 0
        
        result = PerformanceResult(
            metric_name="hash_hit_rate",
            value=hit_rate,
            unit="ratio",
            metadata={"total_lookups": total_lookups, "total_hits": total_hits},
        )
        result.evaluate(self.targets.hash_hit_rate, higher_is_better=True)
        
        return result
    
    def run_full_benchmark(
        self,
        engine,
        evaluator,
        positions: list
    ) -> PerformanceReport:
        """Run complete performance benchmark suite.
        
        Args:
            engine: Chess engine instance.
            evaluator: Neural evaluator instance.
            positions: Test positions.
            
        Returns:
            Complete performance report.
        """
        report = PerformanceReport(test_name="full_performance_benchmark")
        
        if positions:
            # Nodes/sec single thread
            result = self.measure_nodes_per_second(
                engine, positions[0], duration_seconds=2.0, num_threads=1
            )
            report.add_result(result)
            
            # NN eval latency
            result = self.measure_nn_eval_latency(evaluator, positions, num_samples=100)
            report.add_result(result)
            
            # Hash hit rate
            result = self.measure_hash_hit_rate(engine, positions, depth=6)
            report.add_result(result)
        
        report.finalize()
        return report

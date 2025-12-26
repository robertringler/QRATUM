"""PHASE VI: Cognition ↔ Execution Feedback Loop

QRATUM must feel the machine.

Key capabilities:
- Feed runtime telemetry (cache misses, memory pressure, latency) into reasoning
- Allow architectural decisions driven by execution reality, not theory
- Close the loop: learning modifies execution, which modifies learning
- Demonstrate improvement driven by execution feedback alone
"""

import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TelemetryType(Enum):
    """Types of runtime telemetry."""

    CACHE_MISS = "cache_miss"
    MEMORY_PRESSURE = "memory_pressure"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_UTILIZATION = "cpu_utilization"
    CONTEXT_SWITCH = "context_switch"
    PAGE_FAULT = "page_fault"


@dataclass
class TelemetryEvent:
    """A single telemetry event from execution."""

    event_id: str
    telemetry_type: TelemetryType
    value: float
    component: str
    context: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PerformanceProfile:
    """Performance profile of a component."""

    component_id: str
    avg_latency: float  # milliseconds
    p95_latency: float  # 95th percentile
    p99_latency: float  # 99th percentile
    cache_miss_rate: float  # 0.0 to 1.0
    memory_pressure: float  # 0.0 (low) to 1.0 (critical)
    cpu_utilization: float  # 0.0 to 1.0
    bottleneck_score: float  # 0.0 (not bottleneck) to 1.0 (severe bottleneck)


@dataclass
class ArchitecturalDecision:
    """A decision made based on execution feedback."""

    decision_id: str
    description: str
    rationale: str  # Why was this decision made?
    execution_evidence: List[str]  # Telemetry event IDs that support this
    expected_impact: str  # What do we expect to improve?
    implemented: bool = False
    actual_impact: Optional[float] = None  # Measured after implementation
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class FeedbackLoopMetrics:
    """Metrics for the feedback loop itself."""

    loop_iteration: int
    decisions_made: int
    decisions_implemented: int
    avg_decision_impact: float  # Average improvement from decisions
    system_performance: float  # Overall system performance metric
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TelemetryCollector:
    """Collects runtime telemetry from system execution."""

    def __init__(self, window_size: int = 1000):
        """Initialize telemetry collector.

        Args:
            window_size: How many events to keep in memory
        """
        self.events: deque = deque(maxlen=window_size)
        self.event_count = 0

    def record(
        self,
        telemetry_type: TelemetryType,
        value: float,
        component: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Record a telemetry event."""
        event = TelemetryEvent(
            event_id=f"telemetry_{self.event_count}",
            telemetry_type=telemetry_type,
            value=value,
            component=component,
            context=context or {},
        )

        self.events.append(event)
        self.event_count += 1
        return event

    def get_events_by_type(
        self, telemetry_type: TelemetryType, component: Optional[str] = None
    ) -> List[TelemetryEvent]:
        """Get events of a specific type."""
        events = [e for e in self.events if e.telemetry_type == telemetry_type]

        if component:
            events = [e for e in events if e.component == component]

        return events

    def get_recent_events(self, count: int = 100) -> List[TelemetryEvent]:
        """Get most recent events."""
        return list(self.events)[-count:]


class PerformanceAnalyzer:
    """Analyzes telemetry to create performance profiles."""

    @staticmethod
    def create_profile(
        component_id: str, telemetry_collector: TelemetryCollector
    ) -> PerformanceProfile:
        """Create performance profile for a component."""
        # Get latency data
        latency_events = telemetry_collector.get_events_by_type(TelemetryType.LATENCY, component_id)
        latencies = [e.value for e in latency_events]

        if latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95_latency = (
                sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else avg_latency
            )
            p99_latency = (
                sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else avg_latency
            )
        else:
            avg_latency = 0.0
            p95_latency = 0.0
            p99_latency = 0.0

        # Get cache miss rate
        cache_miss_events = telemetry_collector.get_events_by_type(
            TelemetryType.CACHE_MISS, component_id
        )
        cache_miss_rate = (
            statistics.mean([e.value for e in cache_miss_events]) if cache_miss_events else 0.0
        )

        # Get memory pressure
        memory_events = telemetry_collector.get_events_by_type(
            TelemetryType.MEMORY_PRESSURE, component_id
        )
        memory_pressure = (
            statistics.mean([e.value for e in memory_events]) if memory_events else 0.0
        )

        # Get CPU utilization
        cpu_events = telemetry_collector.get_events_by_type(
            TelemetryType.CPU_UTILIZATION, component_id
        )
        cpu_utilization = statistics.mean([e.value for e in cpu_events]) if cpu_events else 0.0

        # Calculate bottleneck score
        # Higher latency, cache misses, and memory pressure = higher bottleneck score
        bottleneck_score = (
            (avg_latency / 1000.0) * 0.4  # Normalize latency
            + cache_miss_rate * 0.3
            + memory_pressure * 0.3
        )
        bottleneck_score = min(1.0, bottleneck_score)

        return PerformanceProfile(
            component_id=component_id,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            cache_miss_rate=cache_miss_rate,
            memory_pressure=memory_pressure,
            cpu_utilization=cpu_utilization,
            bottleneck_score=bottleneck_score,
        )

    @staticmethod
    def identify_bottlenecks(
        profiles: Dict[str, PerformanceProfile], threshold: float = 0.6
    ) -> List[Tuple[str, PerformanceProfile]]:
        """Identify bottleneck components."""
        bottlenecks = [
            (component_id, profile)
            for component_id, profile in profiles.items()
            if profile.bottleneck_score >= threshold
        ]

        # Sort by bottleneck score (worst first)
        bottlenecks.sort(key=lambda x: x[1].bottleneck_score, reverse=True)

        return bottlenecks


class DecisionEngine:
    """Makes architectural decisions based on execution feedback."""

    @staticmethod
    def propose_decisions(
        profiles: Dict[str, PerformanceProfile], bottlenecks: List[Tuple[str, PerformanceProfile]]
    ) -> List[ArchitecturalDecision]:
        """Propose decisions based on performance profiles."""
        decisions = []

        for component_id, profile in bottlenecks:
            # High cache miss rate -> consider data structure change
            if profile.cache_miss_rate > 0.3:
                decisions.append(
                    ArchitecturalDecision(
                        decision_id=f"decision_cache_{component_id}_{datetime.utcnow().timestamp()}",
                        description=f"Optimize data layout for {component_id}",
                        rationale=f"Cache miss rate is {profile.cache_miss_rate:.2%}, indicating poor data locality",
                        execution_evidence=[],  # Would include telemetry event IDs
                        expected_impact="Reduce cache miss rate by 30-50%, improve latency by 20-40%",
                    )
                )

            # High memory pressure -> consider memory optimization
            if profile.memory_pressure > 0.7:
                decisions.append(
                    ArchitecturalDecision(
                        decision_id=f"decision_memory_{component_id}_{datetime.utcnow().timestamp()}",
                        description=f"Reduce memory footprint of {component_id}",
                        rationale=f"Memory pressure is {profile.memory_pressure:.2%}, risking OOM",
                        execution_evidence=[],
                        expected_impact="Reduce memory usage by 20-40%",
                    )
                )

            # High latency -> consider algorithmic improvement
            if profile.avg_latency > 100.0:  # >100ms
                decisions.append(
                    ArchitecturalDecision(
                        decision_id=f"decision_latency_{component_id}_{datetime.utcnow().timestamp()}",
                        description=f"Optimize algorithm for {component_id}",
                        rationale=f"Average latency is {profile.avg_latency:.2f}ms, P99 is {profile.p99_latency:.2f}ms",
                        execution_evidence=[],
                        expected_impact="Reduce latency by 30-60%",
                    )
                )

        return decisions


class ExecutionFeedbackLoop:
    """Closed loop between cognition (reasoning) and execution (performance).

    The system:
    1. Observes its own execution via telemetry
    2. Reasons about performance bottlenecks
    3. Makes architectural decisions
    4. Implements changes
    5. Observes new execution behavior
    6. Repeat

    Success = demonstrable improvement driven by execution feedback alone.
    """

    def __init__(self):
        """Initialize feedback loop."""
        self.telemetry_collector = TelemetryCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.decision_engine = DecisionEngine()

        self.component_profiles: Dict[str, PerformanceProfile] = {}
        self.decisions: Dict[str, ArchitecturalDecision] = {}
        self.metrics_history: List[FeedbackLoopMetrics] = []

        self.iteration = 0

    def record_telemetry(
        self,
        telemetry_type: TelemetryType,
        value: float,
        component: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record runtime telemetry."""
        self.telemetry_collector.record(telemetry_type, value, component, context)

    def run_feedback_iteration(self, components: List[str]) -> Dict[str, Any]:
        """Run one iteration of the feedback loop.

        Returns:
            Summary of the iteration
        """
        self.iteration += 1

        # Step 1: Analyze performance
        profiles = {}
        for component in components:
            profiles[component] = self.performance_analyzer.create_profile(
                component, self.telemetry_collector
            )

        self.component_profiles = profiles

        # Step 2: Identify bottlenecks
        bottlenecks = self.performance_analyzer.identify_bottlenecks(profiles)

        # Step 3: Propose decisions
        new_decisions = self.decision_engine.propose_decisions(profiles, bottlenecks)

        for decision in new_decisions:
            self.decisions[decision.decision_id] = decision

        # Step 4: Calculate metrics
        implemented_decisions = [d for d in self.decisions.values() if d.implemented]
        decisions_with_impact = [d for d in implemented_decisions if d.actual_impact is not None]

        avg_impact = (
            statistics.mean([d.actual_impact for d in decisions_with_impact])
            if decisions_with_impact
            else 0.0
        )

        # System performance = inverse of average bottleneck score
        avg_bottleneck = (
            statistics.mean([p.bottleneck_score for p in profiles.values()]) if profiles else 0.0
        )
        system_performance = 1.0 - avg_bottleneck

        metrics = FeedbackLoopMetrics(
            loop_iteration=self.iteration,
            decisions_made=len(new_decisions),
            decisions_implemented=len(implemented_decisions),
            avg_decision_impact=avg_impact,
            system_performance=system_performance,
        )

        self.metrics_history.append(metrics)

        return {
            "iteration": self.iteration,
            "profiles": {
                cid: {
                    "avg_latency": p.avg_latency,
                    "cache_miss_rate": p.cache_miss_rate,
                    "memory_pressure": p.memory_pressure,
                    "bottleneck_score": p.bottleneck_score,
                }
                for cid, p in profiles.items()
            },
            "bottlenecks": [
                {"component": cid, "score": p.bottleneck_score} for cid, p in bottlenecks
            ],
            "new_decisions": len(new_decisions),
            "system_performance": system_performance,
        }

    def implement_decision(self, decision_id: str, actual_impact: float) -> bool:
        """Mark a decision as implemented and record its impact.

        Args:
            decision_id: ID of decision to implement
            actual_impact: Measured improvement (1.0 = no change, >1.0 = improvement)

        Returns:
            True if successful
        """
        if decision_id not in self.decisions:
            return False

        decision = self.decisions[decision_id]
        decision.implemented = True
        decision.actual_impact = actual_impact

        return True

    def demonstrate_improvement(self) -> Dict[str, Any]:
        """Demonstrate that system improved through feedback alone.

        Returns evidence of improvement over iterations.
        """
        if len(self.metrics_history) < 2:
            return {"improved": False, "reason": "Insufficient iterations", "evidence": None}

        # Compare first and latest iterations
        first = self.metrics_history[0]
        latest = self.metrics_history[-1]

        # Performance improvement
        perf_improvement = latest.system_performance - first.system_performance
        perf_improved = perf_improvement > 0.05  # At least 5% improvement

        # Decision effectiveness
        effective_decisions = latest.avg_decision_impact > 1.05  # At least 5% impact

        # Overall improvement
        improved = perf_improved and effective_decisions

        evidence = {
            "iterations": len(self.metrics_history),
            "initial_performance": first.system_performance,
            "final_performance": latest.system_performance,
            "performance_improvement": perf_improvement,
            "performance_improvement_pct": (perf_improvement / max(first.system_performance, 0.01))
            * 100,
            "total_decisions": latest.decisions_made,
            "implemented_decisions": latest.decisions_implemented,
            "avg_decision_impact": latest.avg_decision_impact,
        }

        return {
            "improved": improved,
            "reason": (
                "Performance improved through execution feedback"
                if improved
                else "Insufficient improvement"
            ),
            "evidence": evidence,
        }

    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """Get current status of feedback loop."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None

        return {
            "iteration": self.iteration,
            "telemetry_events": len(self.telemetry_collector.events),
            "component_profiles": len(self.component_profiles),
            "total_decisions": len(self.decisions),
            "implemented_decisions": sum(1 for d in self.decisions.values() if d.implemented),
            "current_performance": latest_metrics.system_performance if latest_metrics else 0.0,
            "improvement_demonstrated": self.demonstrate_improvement()["improved"],
        }

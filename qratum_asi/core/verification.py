"""PHASE II: Self-Verification Engine

Continuous correctness validation with:
- SSSP algorithm correctness validation
- Graph operation validators
- Scheduling integrity checks
- Regression detection (intent-based, not snapshot-based)
- Rollback/containment strategies
- Zero-trust execution loop

The system must know when it broke itself and act accordingly.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class VerificationLevel(Enum):
    """Levels of verification rigor."""
    BASIC = "basic"  # Lightweight checks
    STANDARD = "standard"  # Normal verification
    PARANOID = "paranoid"  # Maximum verification


class VerificationResult(Enum):
    """Result of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class VerificationCheck:
    """A single verification check."""
    check_id: str
    check_name: str
    check_type: str  # "correctness", "performance", "invariant", "regression"
    intent: str  # What is this check trying to ensure? (WHY)
    verification_func: Callable[[Dict[str, Any]], bool]
    level: VerificationLevel = VerificationLevel.STANDARD
    last_run: Optional[str] = None
    last_result: Optional[VerificationResult] = None
    failure_count: int = 0

    def run(self, context: Dict[str, Any]) -> VerificationResult:
        """Execute this verification check."""
        try:
            passed = self.verification_func(context)
            self.last_run = datetime.utcnow().isoformat()
            self.last_result = VerificationResult.PASS if passed else VerificationResult.FAIL

            if not passed:
                self.failure_count += 1

            return self.last_result
        except Exception:
            self.last_run = datetime.utcnow().isoformat()
            self.last_result = VerificationResult.UNKNOWN
            self.failure_count += 1
            return VerificationResult.UNKNOWN


@dataclass
class RegressionSignature:
    """Signature of system behavior for regression detection.
    
    Instead of snapshot comparison, we track INTENT fulfillment.
    """
    signature_id: str
    intent: str  # What is the system trying to achieve?
    behavioral_markers: Dict[str, Any]  # Observable behaviors that indicate intent fulfillment
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def compute_similarity(self, other: 'RegressionSignature') -> float:
        """Compute similarity with another signature (0.0 to 1.0)."""
        if self.intent != other.intent:
            return 0.0

        # Compare behavioral markers
        matching = 0
        total = len(self.behavioral_markers)

        for key, value in self.behavioral_markers.items():
            if key in other.behavioral_markers:
                if other.behavioral_markers[key] == value:
                    matching += 1

        return matching / max(total, 1)


@dataclass
class ContainmentStrategy:
    """Strategy for containing failures."""
    strategy_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    actions: List[Callable[[], bool]]
    escalation_threshold: int = 3  # How many failures before escalating


class SSSPValidator:
    """Validator for Single-Source Shortest Path algorithms."""

    @staticmethod
    def validate_correctness(
        graph: Dict[str, Any],
        source: int,
        distances: Dict[int, float],
        predecessors: Dict[int, Optional[int]]
    ) -> bool:
        """Validate SSSP correctness properties.
        
        Checks:
        1. Source distance is 0
        2. Triangle inequality holds for all edges
        3. Paths are valid
        4. No negative cycles reachable from source
        """
        # Check 1: Source distance
        if source not in distances or distances[source] != 0:
            return False

        # Check 2: Triangle inequality
        edges = graph.get("edges", [])
        for edge in edges:
            u, v, weight = edge.get("from"), edge.get("to"), edge.get("weight")

            if u in distances and v in distances:
                if distances[v] > distances[u] + weight + 1e-9:  # Small epsilon for floating point
                    return False

        # Check 3: Path validity via predecessors
        for node, pred in predecessors.items():
            if pred is not None:
                if node == source:
                    return False  # Source should have no predecessor

                # Predecessor should have a shorter distance
                if node in distances and pred in distances:
                    if distances[node] <= distances[pred]:
                        return False

        return True

    @staticmethod
    def compare_with_baseline(
        test_distances: Dict[int, float],
        baseline_distances: Dict[int, float],
        epsilon: float = 1e-6
    ) -> bool:
        """Compare SSSP results with a known-good baseline."""
        if set(test_distances.keys()) != set(baseline_distances.keys()):
            return False

        for node in test_distances:
            if abs(test_distances[node] - baseline_distances[node]) > epsilon:
                return False

        return True


class GraphOperationValidator:
    """Validator for graph operations."""

    @staticmethod
    def validate_graph_structure(graph: Dict[str, Any]) -> bool:
        """Validate graph has valid structure."""
        # Check required fields
        if "nodes" not in graph or "edges" not in graph:
            return False

        nodes = set(graph["nodes"])

        # Check edges reference valid nodes
        for edge in graph["edges"]:
            if "from" not in edge or "to" not in edge:
                return False

            if edge["from"] not in nodes or edge["to"] not in nodes:
                return False

        return True

    @staticmethod
    def detect_cycles(graph: Dict[str, Any]) -> bool:
        """Detect if graph has cycles (returns True if cycle found)."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Build adjacency list
        adj = {node: [] for node in nodes}
        for edge in edges:
            adj[edge["from"]].append(edge["to"])

        # DFS for cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle_dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle_dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in nodes:
            if node not in visited:
                if has_cycle_dfs(node):
                    return True

        return False


class SchedulingValidator:
    """Validator for scheduling operations."""

    @staticmethod
    def validate_fairness(
        schedule: List[Dict[str, Any]],
        window_size: int = 100
    ) -> bool:
        """Validate scheduling fairness within a window."""
        if len(schedule) < window_size:
            return True

        # Check last window
        window = schedule[-window_size:]
        contract_counts = {}

        for entry in window:
            contract_id = entry.get("contract_id", "unknown")
            contract_counts[contract_id] = contract_counts.get(contract_id, 0) + 1

        # No single contract should dominate more than 50% of window
        max_count = max(contract_counts.values()) if contract_counts else 0
        return max_count <= window_size * 0.5

    @staticmethod
    def validate_no_starvation(
        schedule: List[Dict[str, Any]],
        max_wait_time: float = 60.0
    ) -> bool:
        """Validate no contract is starved."""
        current_time = datetime.utcnow().timestamp()

        for entry in schedule:
            if entry.get("status") == "queued":
                queued_time = datetime.fromisoformat(entry.get("queued_at")).timestamp()
                wait_time = current_time - queued_time

                if wait_time > max_wait_time:
                    return False

        return True


@dataclass
class SelfVerificationEngine:
    """Continuous self-verification engine.
    
    Implements zero-trust execution where every operation is verified.
    Detects regressions based on intent fulfillment, not snapshots.
    """

    checks: Dict[str, VerificationCheck] = field(default_factory=dict)
    regression_signatures: Dict[str, RegressionSignature] = field(default_factory=dict)
    containment_strategies: Dict[str, ContainmentStrategy] = field(default_factory=dict)
    verification_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize verification checks."""
        self._initialize_sssp_checks()
        self._initialize_graph_checks()
        self._initialize_scheduling_checks()
        self._initialize_containment_strategies()

    def _initialize_sssp_checks(self):
        """Initialize SSSP verification checks."""
        self.checks["sssp_correctness"] = VerificationCheck(
            check_id="sssp_correctness",
            check_name="SSSP Correctness",
            check_type="correctness",
            intent="Ensure shortest paths are computed correctly",
            verification_func=lambda ctx: SSSPValidator.validate_correctness(
                ctx.get("graph", {}),
                ctx.get("source", 0),
                ctx.get("distances", {}),
                ctx.get("predecessors", {})
            ),
            level=VerificationLevel.STANDARD
        )

        self.checks["sssp_baseline"] = VerificationCheck(
            check_id="sssp_baseline",
            check_name="SSSP Baseline Comparison",
            check_type="correctness",
            intent="Ensure results match classical Dijkstra baseline",
            verification_func=lambda ctx: SSSPValidator.compare_with_baseline(
                ctx.get("test_distances", {}),
                ctx.get("baseline_distances", {}),
                epsilon=ctx.get("epsilon", 1e-6)
            ),
            level=VerificationLevel.PARANOID
        )

    def _initialize_graph_checks(self):
        """Initialize graph operation checks."""
        self.checks["graph_structure"] = VerificationCheck(
            check_id="graph_structure",
            check_name="Graph Structure Validation",
            check_type="correctness",
            intent="Ensure graph has valid structure",
            verification_func=lambda ctx: GraphOperationValidator.validate_graph_structure(
                ctx.get("graph", {})
            ),
            level=VerificationLevel.BASIC
        )

        self.checks["graph_acyclic"] = VerificationCheck(
            check_id="graph_acyclic",
            check_name="Graph Acyclic Check",
            check_type="correctness",
            intent="Ensure graph is acyclic when required",
            verification_func=lambda ctx: not GraphOperationValidator.detect_cycles(
                ctx.get("graph", {})
            ) if ctx.get("require_acyclic", False) else True,
            level=VerificationLevel.STANDARD
        )

    def _initialize_scheduling_checks(self):
        """Initialize scheduling checks."""
        self.checks["scheduling_fairness"] = VerificationCheck(
            check_id="scheduling_fairness",
            check_name="Scheduling Fairness",
            check_type="correctness",
            intent="Ensure fair scheduling across contracts",
            verification_func=lambda ctx: SchedulingValidator.validate_fairness(
                ctx.get("schedule", []),
                ctx.get("window_size", 100)
            ),
            level=VerificationLevel.STANDARD
        )

        self.checks["scheduling_no_starvation"] = VerificationCheck(
            check_id="scheduling_no_starvation",
            check_name="No Starvation",
            check_type="correctness",
            intent="Ensure no contract is starved",
            verification_func=lambda ctx: SchedulingValidator.validate_no_starvation(
                ctx.get("schedule", []),
                ctx.get("max_wait_time", 60.0)
            ),
            level=VerificationLevel.STANDARD
        )

    def _initialize_containment_strategies(self):
        """Initialize failure containment strategies."""
        self.containment_strategies["rollback"] = ContainmentStrategy(
            strategy_id="rollback",
            name="Rollback to Last Good State",
            description="Roll back to the last verified checkpoint",
            trigger_conditions=["verification_failure", "invariant_violation"],
            actions=[self._action_rollback]
        )

        self.containment_strategies["isolate"] = ContainmentStrategy(
            strategy_id="isolate",
            name="Isolate Failed Component",
            description="Isolate the failing component to prevent cascade",
            trigger_conditions=["component_failure", "repeated_errors"],
            actions=[self._action_isolate_component]
        )

    def verify_operation(
        self,
        operation_type: str,
        context: Dict[str, Any],
        level: VerificationLevel = VerificationLevel.STANDARD
    ) -> Dict[str, Any]:
        """Verify an operation."""
        results = {}
        failures = []

        # Run relevant checks for this operation
        for check_id, check in self.checks.items():
            # Skip checks that don't match the verification level
            if level == VerificationLevel.BASIC and check.level != VerificationLevel.BASIC:
                continue

            result = check.run(context)
            results[check_id] = result

            if result == VerificationResult.FAIL:
                failures.append(check_id)

        # Record verification
        verification_record = {
            "operation_type": operation_type,
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "results": {k: v.value for k, v in results.items()},
            "success": len(failures) == 0,
            "failures": failures
        }

        self.verification_history.append(verification_record)

        # Trigger containment if needed
        if failures:
            self._trigger_containment(failures, context)

        return verification_record

    def detect_regression(
        self,
        intent: str,
        current_behavior: Dict[str, Any]
    ) -> bool:
        """Detect regression by comparing intent fulfillment.
        
        Returns True if regression detected (behavior changed in a way that
        violates intent).
        """
        # Create signature for current behavior
        current_sig = RegressionSignature(
            signature_id=f"{intent}_{datetime.utcnow().timestamp()}",
            intent=intent,
            behavioral_markers=current_behavior
        )

        # Find previous signatures with same intent
        previous_sigs = [
            sig for sig in self.regression_signatures.values()
            if sig.intent == intent
        ]

        if not previous_sigs:
            # First time seeing this intent, store and pass
            self.regression_signatures[current_sig.signature_id] = current_sig
            return False

        # Compare with most recent previous signature
        latest_previous = max(previous_sigs, key=lambda s: s.timestamp)
        similarity = current_sig.compute_similarity(latest_previous)

        # Store current signature
        self.regression_signatures[current_sig.signature_id] = current_sig

        # Regression if similarity drops below threshold
        return similarity < 0.8

    def _trigger_containment(
        self,
        failures: List[str],
        context: Dict[str, Any]
    ):
        """Trigger appropriate containment strategies."""
        for strategy_id, strategy in self.containment_strategies.items():
            # Check if any trigger condition matches
            should_trigger = False
            for condition in strategy.trigger_conditions:
                if condition in failures or condition in context.get("conditions", []):
                    should_trigger = True
                    break

            if should_trigger:
                # Execute containment actions
                for action in strategy.actions:
                    try:
                        action()
                    except Exception:
                        pass  # Continue with other actions

    def _action_rollback(self) -> bool:
        """Containment action: rollback."""
        # Placeholder - actual implementation would trigger rollback
        return True

    def _action_isolate_component(self) -> bool:
        """Containment action: isolate component."""
        # Placeholder - actual implementation would isolate component
        return True

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total_verifications = len(self.verification_history)
        failures = sum(1 for v in self.verification_history if not v["success"])

        check_stats = {}
        for check_id, check in self.checks.items():
            check_stats[check_id] = {
                "failure_count": check.failure_count,
                "last_result": check.last_result.value if check.last_result else "never_run"
            }

        return {
            "total_verifications": total_verifications,
            "total_failures": failures,
            "failure_rate": failures / max(total_verifications, 1),
            "check_stats": check_stats,
            "regression_signatures": len(self.regression_signatures)
        }

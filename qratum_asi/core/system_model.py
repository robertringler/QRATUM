"""PHASE I: System Self-Model Construction

Build a formal internal representation of QRATUM/QRADLE that enables:
- Graph execution tracking
- Memory layout analysis
- Scheduling state monitoring
- Failure mode detection
- Invariant encoding as first-class objects
- Auto-update mechanism when system changes

This is the foundation for system-level reasoning and self-improvement.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ComponentType(Enum):
    """Types of system components."""

    GRAPH_EXECUTOR = "graph_executor"
    MEMORY_MANAGER = "memory_manager"
    SCHEDULER = "scheduler"
    CONTRACT_ENGINE = "contract_engine"
    MERKLE_CHAIN = "merkle_chain"
    INVARIANT_CHECKER = "invariant_checker"
    ROLLBACK_MANAGER = "rollback_manager"


class FailureMode(Enum):
    """Known failure modes of the system."""

    MEMORY_EXHAUSTION = "memory_exhaustion"
    DEADLOCK = "deadlock"
    INVARIANT_VIOLATION = "invariant_violation"
    GRAPH_CYCLE = "graph_cycle"
    CONTRACT_TIMEOUT = "contract_timeout"
    MERKLE_CORRUPTION = "merkle_corruption"
    ROLLBACK_FAILURE = "rollback_failure"


@dataclass
class InvariantModel:
    """First-class representation of a system invariant.

    Encodes not just WHAT must hold, but WHY it matters.
    """

    invariant_id: str
    name: str
    description: str
    rationale: str  # WHY this invariant exists
    criticality: str  # ROUTINE, ELEVATED, SENSITIVE, CRITICAL, EXISTENTIAL
    validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    last_validated: Optional[str] = None
    violation_count: int = 0

    def validate(self, system_state: Dict[str, Any]) -> bool:
        """Validate this invariant against current system state."""
        if self.validation_func is None:
            return True

        try:
            result = self.validation_func(system_state)
            self.last_validated = datetime.utcnow().isoformat()
            if not result:
                self.violation_count += 1
            return result
        except Exception:
            self.violation_count += 1
            return False


@dataclass
class ComponentModel:
    """Model of a system component."""

    component_id: str
    component_type: ComponentType
    state: Dict[str, Any]
    dependencies: List[str]  # IDs of components this depends on
    invariants: List[str]  # IDs of invariants this component must maintain
    failure_modes: List[FailureMode]
    performance_bounds: Dict[str, float]  # e.g., {"max_latency_ms": 100}
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_state_hash(self) -> str:
        """Compute hash of component state for change detection."""
        state_json = json.dumps(self.state, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()


@dataclass
class GraphExecutionModel:
    """Model of graph execution behavior."""

    graph_id: str
    node_count: int
    edge_count: int
    execution_pattern: str  # e.g., "BFS", "DFS", "parallel"
    memory_footprint: int  # bytes
    avg_execution_time: float  # seconds
    failure_rate: float  # 0.0 to 1.0
    recent_executions: List[Dict[str, Any]] = field(default_factory=list)

    def add_execution(self, execution_data: Dict[str, Any]):
        """Record an execution for pattern analysis."""
        self.recent_executions.append(
            {**execution_data, "timestamp": datetime.utcnow().isoformat()}
        )
        # Keep only last 100 executions
        if len(self.recent_executions) > 100:
            self.recent_executions = self.recent_executions[-100:]

        # Update statistics
        if "execution_time" in execution_data:
            # Exponential moving average
            alpha = 0.1
            self.avg_execution_time = (
                alpha * execution_data["execution_time"] + (1 - alpha) * self.avg_execution_time
            )


@dataclass
class MemoryLayoutModel:
    """Model of memory usage and layout."""

    total_allocated: int  # bytes
    peak_allocated: int  # bytes
    allocation_patterns: Dict[str, int]  # component -> bytes
    fragmentation_ratio: float  # 0.0 to 1.0
    pressure_level: str  # "low", "medium", "high", "critical"

    def update_pressure_level(self):
        """Update pressure level based on allocation."""
        usage_ratio = self.total_allocated / max(self.peak_allocated, 1)
        if usage_ratio > 0.95:
            self.pressure_level = "critical"
        elif usage_ratio > 0.80:
            self.pressure_level = "high"
        elif usage_ratio > 0.60:
            self.pressure_level = "medium"
        else:
            self.pressure_level = "low"


@dataclass
class SchedulingModel:
    """Model of scheduling behavior."""

    active_contracts: int
    queued_contracts: int
    average_wait_time: float  # seconds
    throughput: float  # contracts/second
    scheduling_policy: str  # e.g., "FIFO", "priority", "fair"
    resource_utilization: Dict[str, float]  # resource -> utilization (0.0-1.0)


@dataclass
class FailureModeModel:
    """Model of a failure mode."""

    failure_mode: FailureMode
    probability: float  # Estimated probability of occurrence
    impact_severity: str  # "low", "medium", "high", "critical"
    detection_method: str  # How we detect this failure
    mitigation_strategy: str  # How we handle/prevent this failure
    occurrence_count: int = 0
    last_occurred: Optional[str] = None

    def record_occurrence(self):
        """Record that this failure occurred."""
        self.occurrence_count += 1
        self.last_occurred = datetime.utcnow().isoformat()


class QRATUMSystemModel:
    """Formal internal representation of the QRATUM/QRADLE system.

    This is the system's understanding of itself - enabling:
    - Self-reasoning about execution
    - Self-verification
    - Self-improvement decisions based on reality
    """

    def __init__(self):
        """Initialize the system self-model."""
        self.components: Dict[str, ComponentModel] = {}
        self.invariants: Dict[str, InvariantModel] = {}
        self.graph_models: Dict[str, GraphExecutionModel] = {}
        self.memory_model: MemoryLayoutModel = MemoryLayoutModel(
            total_allocated=0,
            peak_allocated=0,
            allocation_patterns={},
            fragmentation_ratio=0.0,
            pressure_level="low",
        )
        self.scheduling_model: SchedulingModel = SchedulingModel(
            active_contracts=0,
            queued_contracts=0,
            average_wait_time=0.0,
            throughput=0.0,
            scheduling_policy="FIFO",
            resource_utilization={},
        )
        self.failure_modes: Dict[FailureMode, FailureModeModel] = {}
        self.model_version: int = 1
        self.last_updated: str = datetime.utcnow().isoformat()

        # Initialize with known failure modes
        self._initialize_failure_modes()

        # Initialize with core invariants
        self._initialize_core_invariants()

    def _initialize_failure_modes(self):
        """Initialize known failure modes."""
        self.failure_modes[FailureMode.MEMORY_EXHAUSTION] = FailureModeModel(
            failure_mode=FailureMode.MEMORY_EXHAUSTION,
            probability=0.05,
            impact_severity="critical",
            detection_method="memory_pressure_monitoring",
            mitigation_strategy="rollback_to_checkpoint",
        )

        self.failure_modes[FailureMode.INVARIANT_VIOLATION] = FailureModeModel(
            failure_mode=FailureMode.INVARIANT_VIOLATION,
            probability=0.01,
            impact_severity="critical",
            detection_method="continuous_invariant_validation",
            mitigation_strategy="halt_and_alert",
        )

        self.failure_modes[FailureMode.GRAPH_CYCLE] = FailureModeModel(
            failure_mode=FailureMode.GRAPH_CYCLE,
            probability=0.02,
            impact_severity="high",
            detection_method="cycle_detection_algorithm",
            mitigation_strategy="reject_cyclic_operations",
        )

    def _initialize_core_invariants(self):
        """Initialize core system invariants as first-class objects."""
        # Invariant 1: Human Oversight
        self.invariants["human_oversight"] = InvariantModel(
            invariant_id="human_oversight",
            name="Human Oversight Requirement",
            description="Sensitive operations require human authorization",
            rationale="Ensures human control over critical system changes",
            criticality="EXISTENTIAL",
        )

        # Invariant 2: Merkle Chain Integrity
        self.invariants["merkle_integrity"] = InvariantModel(
            invariant_id="merkle_integrity",
            name="Merkle Chain Integrity",
            description="All events must be cryptographically chained",
            rationale="Provides tamper-evident audit trail for all operations",
            criticality="CRITICAL",
        )

        # Invariant 3: Contract Immutability
        self.invariants["contract_immutability"] = InvariantModel(
            invariant_id="contract_immutability",
            name="Contract Immutability",
            description="Executed contracts cannot be retroactively altered",
            rationale="Ensures reproducibility and prevents history rewriting",
            criticality="CRITICAL",
        )

        # Invariant 8: Determinism Guarantee
        self.invariants["determinism"] = InvariantModel(
            invariant_id="determinism",
            name="Determinism Guarantee",
            description="Same inputs must produce same outputs",
            rationale="Enables verification, certification, and trust",
            criticality="CRITICAL",
        )

    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        initial_state: Dict[str, Any],
        dependencies: List[str],
        invariants: List[str],
        failure_modes: List[FailureMode],
        performance_bounds: Dict[str, float],
    ) -> ComponentModel:
        """Register a component in the system model."""
        component = ComponentModel(
            component_id=component_id,
            component_type=component_type,
            state=initial_state,
            dependencies=dependencies,
            invariants=invariants,
            failure_modes=failure_modes,
            performance_bounds=performance_bounds,
        )

        self.components[component_id] = component
        self._mark_model_updated()
        return component

    def update_component_state(self, component_id: str, new_state: Dict[str, Any]):
        """Update the state of a component."""
        if component_id not in self.components:
            raise ValueError(f"Component not found: {component_id}")

        component = self.components[component_id]
        old_hash = component.get_state_hash()
        component.state = new_state
        component.last_updated = datetime.utcnow().isoformat()
        new_hash = component.get_state_hash()

        # If state changed, mark model as updated
        if old_hash != new_hash:
            self._mark_model_updated()

    def register_graph_execution(
        self, graph_id: str, node_count: int, edge_count: int, execution_data: Dict[str, Any]
    ):
        """Register a graph execution for pattern learning."""
        if graph_id not in self.graph_models:
            self.graph_models[graph_id] = GraphExecutionModel(
                graph_id=graph_id,
                node_count=node_count,
                edge_count=edge_count,
                execution_pattern=execution_data.get("pattern", "unknown"),
                memory_footprint=execution_data.get("memory_footprint", 0),
                avg_execution_time=execution_data.get("execution_time", 0.0),
                failure_rate=0.0,
            )

        self.graph_models[graph_id].add_execution(execution_data)

    def update_memory_model(self, total_allocated: int, allocation_patterns: Dict[str, int]):
        """Update memory model with current allocation."""
        self.memory_model.total_allocated = total_allocated
        self.memory_model.allocation_patterns = allocation_patterns

        if total_allocated > self.memory_model.peak_allocated:
            self.memory_model.peak_allocated = total_allocated

        self.memory_model.update_pressure_level()
        self._mark_model_updated()

    def update_scheduling_model(
        self,
        active_contracts: int,
        queued_contracts: int,
        average_wait_time: float,
        throughput: float,
    ):
        """Update scheduling model."""
        self.scheduling_model.active_contracts = active_contracts
        self.scheduling_model.queued_contracts = queued_contracts
        self.scheduling_model.average_wait_time = average_wait_time
        self.scheduling_model.throughput = throughput
        self._mark_model_updated()

    def record_failure(self, failure_mode: FailureMode):
        """Record occurrence of a failure mode."""
        if failure_mode in self.failure_modes:
            self.failure_modes[failure_mode].record_occurrence()

    def validate_all_invariants(self) -> Dict[str, bool]:
        """Validate all invariants against current system state.

        Returns:
            Dictionary mapping invariant_id to validation result
        """
        system_state = self.get_system_state()
        results = {}

        for inv_id, invariant in self.invariants.items():
            results[inv_id] = invariant.validate(system_state)

        return results

    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state for analysis."""
        return {
            "components": {cid: comp.state for cid, comp in self.components.items()},
            "memory": {
                "total_allocated": self.memory_model.total_allocated,
                "pressure_level": self.memory_model.pressure_level,
            },
            "scheduling": {
                "active_contracts": self.scheduling_model.active_contracts,
                "queued_contracts": self.scheduling_model.queued_contracts,
            },
            "model_version": self.model_version,
        }

    def get_failure_predictions(self) -> List[Dict[str, Any]]:
        """Predict likely failures based on current state."""
        predictions = []

        # Check memory pressure
        if self.memory_model.pressure_level in ["high", "critical"]:
            predictions.append(
                {
                    "failure_mode": FailureMode.MEMORY_EXHAUSTION,
                    "probability": 0.7 if self.memory_model.pressure_level == "critical" else 0.3,
                    "reason": f"Memory pressure is {self.memory_model.pressure_level}",
                }
            )

        # Check scheduling backlog
        if self.scheduling_model.queued_contracts > 100:
            predictions.append(
                {
                    "failure_mode": FailureMode.DEADLOCK,
                    "probability": 0.4,
                    "reason": f"Large queue backlog: {self.scheduling_model.queued_contracts}",
                }
            )

        return predictions

    def _mark_model_updated(self):
        """Mark that the model has been updated."""
        self.model_version += 1
        self.last_updated = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Export system model as machine-readable dictionary."""
        return {
            "model_version": self.model_version,
            "last_updated": self.last_updated,
            "components": {
                cid: {
                    "type": comp.component_type.value,
                    "state_hash": comp.get_state_hash(),
                    "dependencies": comp.dependencies,
                    "invariants": comp.invariants,
                    "failure_modes": [fm.value for fm in comp.failure_modes],
                }
                for cid, comp in self.components.items()
            },
            "invariants": {
                inv_id: {
                    "name": inv.name,
                    "criticality": inv.criticality,
                    "violation_count": inv.violation_count,
                }
                for inv_id, inv in self.invariants.items()
            },
            "memory": {
                "total_allocated": self.memory_model.total_allocated,
                "peak_allocated": self.memory_model.peak_allocated,
                "pressure_level": self.memory_model.pressure_level,
                "fragmentation_ratio": self.memory_model.fragmentation_ratio,
            },
            "scheduling": {
                "active_contracts": self.scheduling_model.active_contracts,
                "queued_contracts": self.scheduling_model.queued_contracts,
                "average_wait_time": self.scheduling_model.average_wait_time,
                "throughput": self.scheduling_model.throughput,
            },
            "failure_modes": {
                fm.value: {
                    "probability": model.probability,
                    "impact_severity": model.impact_severity,
                    "occurrence_count": model.occurrence_count,
                }
                for fm, model in self.failure_modes.items()
            },
        }

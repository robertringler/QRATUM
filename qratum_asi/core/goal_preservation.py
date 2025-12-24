"""PHASE III: Goal Preservation Under Change

Ensure QRATUM can change implementation without changing purpose.

Key capabilities:
- Replace static rules with constraint rationales
- Encode "why this matters" alongside "what must hold"
- Test recursive refactors for goal preservation
- Collect evidence of goal stability under architectural mutation
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GoalCategory(Enum):
    """Categories of system goals."""
    SAFETY = "safety"  # System must be safe
    CORRECTNESS = "correctness"  # System must be correct
    PERFORMANCE = "performance"  # System must be performant
    AUDITABILITY = "auditability"  # System must be auditable
    CONTROLLABILITY = "controllability"  # System must be controllable


class ConstraintType(Enum):
    """Types of constraints."""
    INVARIANT = "invariant"  # Must always hold
    PREFERENCE = "preference"  # Should hold when possible
    OPTIMIZATION = "optimization"  # Improve when possible


@dataclass
class Rationale:
    """Rationale explaining WHY a constraint exists."""
    rationale_id: str
    constraint_id: str
    reason: str  # Human-readable explanation
    goal_category: GoalCategory  # Which goal does this serve?
    consequences_if_violated: str  # What happens if we don't maintain this?
    alternatives_considered: List[str]  # What else did we consider?
    decision_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Constraint:
    """A constraint with its rationale.
    
    Not just WHAT must hold, but WHY it matters.
    """
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    rationale: Rationale
    validation_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    priority: int = 1  # Higher = more important

    def get_purpose_hash(self) -> str:
        """Hash of the constraint's PURPOSE (not implementation)."""
        purpose = {
            "name": self.name,
            "goal_category": self.rationale.goal_category.value,
            "reason": self.rationale.reason
        }
        return hashlib.sha256(json.dumps(purpose, sort_keys=True).encode()).hexdigest()


@dataclass
class Goal:
    """A high-level goal of the system."""
    goal_id: str
    name: str
    description: str
    category: GoalCategory
    success_criteria: List[str]  # How do we know if goal is achieved?
    related_constraints: List[str]  # Constraint IDs that support this goal
    measurement_func: Optional[Callable[[Dict[str, Any]], float]] = None  # Returns 0.0-1.0

    def measure_achievement(self, system_state: Dict[str, Any]) -> float:
        """Measure how well this goal is currently achieved (0.0 to 1.0)."""
        if self.measurement_func is None:
            return 1.0  # Assume achieved if no measurement

        try:
            return max(0.0, min(1.0, self.measurement_func(system_state)))
        except Exception:
            return 0.0


@dataclass
class ArchitecturalChange:
    """Represents a change to system architecture."""
    change_id: str
    description: str
    affected_components: List[str]
    implementation_before: str  # Hash of implementation before change
    implementation_after: str  # Hash of implementation after change
    purpose_before: str  # Hash of purpose before change
    purpose_after: str  # Hash of purpose after change
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def preserves_purpose(self) -> bool:
        """Check if this change preserves purpose."""
        return self.purpose_before == self.purpose_after


@dataclass
class GoalPreservationTest:
    """Test to verify goal preservation under change."""
    test_id: str
    test_name: str
    goal_id: str
    test_scenario: str
    validation_func: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    # Takes (state_before, state_after) and returns True if goal preserved

    def run(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> bool:
        """Run the preservation test."""
        try:
            return self.validation_func(state_before, state_after)
        except Exception:
            return False


class GoalPreservationEngine:
    """Engine for ensuring goals are preserved during system changes.
    
    This is critical for recursive self-modification - the system must
    maintain its purpose even as implementation changes.
    """

    def __init__(self):
        """Initialize goal preservation engine."""
        self.goals: Dict[str, Goal] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.rationales: Dict[str, Rationale] = {}
        self.changes: List[ArchitecturalChange] = []
        self.preservation_tests: Dict[str, GoalPreservationTest] = {}

        # Initialize core goals
        self._initialize_core_goals()

        # Initialize core constraints with rationales
        self._initialize_core_constraints()

        # Initialize preservation tests
        self._initialize_preservation_tests()

    def _initialize_core_goals(self):
        """Initialize core system goals."""
        # Goal 1: System Safety
        self.goals["safety"] = Goal(
            goal_id="safety",
            name="System Safety",
            description="System must not cause harm and must prevent harmful operations",
            category=GoalCategory.SAFETY,
            success_criteria=[
                "No unauthorized modifications to safety constraints",
                "Human oversight maintained for critical operations",
                "Rollback capability always available"
            ],
            related_constraints=[],
            measurement_func=self._measure_safety_goal
        )

        # Goal 2: Correctness
        self.goals["correctness"] = Goal(
            goal_id="correctness",
            name="Correctness",
            description="System must produce correct results",
            category=GoalCategory.CORRECTNESS,
            success_criteria=[
                "All outputs match specification",
                "Determinism maintained",
                "Invariants never violated"
            ],
            related_constraints=[],
            measurement_func=self._measure_correctness_goal
        )

        # Goal 3: Auditability
        self.goals["auditability"] = Goal(
            goal_id="auditability",
            name="Auditability",
            description="All operations must be traceable and verifiable",
            category=GoalCategory.AUDITABILITY,
            success_criteria=[
                "Merkle chain maintained",
                "All events logged",
                "Proofs available for all operations"
            ],
            related_constraints=[],
            measurement_func=self._measure_auditability_goal
        )

        # Goal 4: Controllability
        self.goals["controllability"] = Goal(
            goal_id="controllability",
            name="Controllability",
            description="Humans must maintain control over the system",
            category=GoalCategory.CONTROLLABILITY,
            success_criteria=[
                "Human authorization enforced for sensitive ops",
                "Shutdown capability maintained",
                "Monitoring active and accessible"
            ],
            related_constraints=[],
            measurement_func=self._measure_controllability_goal
        )

    def _initialize_core_constraints(self):
        """Initialize core constraints with rationales."""
        # Constraint: Human Oversight
        rationale_oversight = Rationale(
            rationale_id="rationale_human_oversight",
            constraint_id="constraint_human_oversight",
            reason="Humans must maintain final authority over critical system changes",
            goal_category=GoalCategory.CONTROLLABILITY,
            consequences_if_violated=(
                "System could make autonomous decisions without human input, "
                "potentially leading to unintended behaviors or harmful outcomes"
            ),
            alternatives_considered=[
                "Fully autonomous operation (rejected: too risky)",
                "Post-hoc review only (rejected: insufficient control)"
            ]
        )

        self.rationales["rationale_human_oversight"] = rationale_oversight

        self.constraints["constraint_human_oversight"] = Constraint(
            constraint_id="constraint_human_oversight",
            name="Human Oversight Requirement",
            description="Sensitive operations require human authorization",
            constraint_type=ConstraintType.INVARIANT,
            rationale=rationale_oversight,
            priority=10  # Highest priority
        )

        # Constraint: Determinism
        rationale_determinism = Rationale(
            rationale_id="rationale_determinism",
            constraint_id="constraint_determinism",
            reason=(
                "Determinism enables verification, debugging, and certification. "
                "Same inputs must produce same outputs for trust and reproducibility."
            ),
            goal_category=GoalCategory.CORRECTNESS,
            consequences_if_violated=(
                "System becomes non-verifiable, certification impossible, "
                "debugging extremely difficult, trust undermined"
            ),
            alternatives_considered=[
                "Probabilistic execution (rejected: breaks verification)",
                "Logged randomness (considered: complex, still impacts certification)"
            ]
        )

        self.rationales["rationale_determinism"] = rationale_determinism

        self.constraints["constraint_determinism"] = Constraint(
            constraint_id="constraint_determinism",
            name="Determinism Guarantee",
            description="Same inputs must produce same outputs",
            constraint_type=ConstraintType.INVARIANT,
            rationale=rationale_determinism,
            priority=10
        )

        # Constraint: Merkle Chain Integrity
        rationale_merkle = Rationale(
            rationale_id="rationale_merkle",
            constraint_id="constraint_merkle",
            reason=(
                "Cryptographic chaining of events provides tamper-evident audit trail. "
                "Essential for trust, compliance, and post-hoc analysis."
            ),
            goal_category=GoalCategory.AUDITABILITY,
            consequences_if_violated=(
                "Audit trail could be tampered with, compliance impossible, "
                "post-incident analysis unreliable"
            ),
            alternatives_considered=[
                "Database logs only (rejected: mutable)",
                "Blockchain (considered: too heavyweight for internal use)"
            ]
        )

        self.rationales["rationale_merkle"] = rationale_merkle

        self.constraints["constraint_merkle"] = Constraint(
            constraint_id="constraint_merkle",
            name="Merkle Chain Integrity",
            description="All events cryptographically chained",
            constraint_type=ConstraintType.INVARIANT,
            rationale=rationale_merkle,
            priority=9
        )

    def _initialize_preservation_tests(self):
        """Initialize goal preservation tests."""
        # Test: Safety preserved during refactor
        self.preservation_tests["safety_refactor"] = GoalPreservationTest(
            test_id="safety_refactor",
            test_name="Safety Preserved During Refactor",
            goal_id="safety",
            test_scenario="Architectural refactoring",
            validation_func=self._validate_safety_preserved
        )

        # Test: Correctness preserved during optimization
        self.preservation_tests["correctness_optimization"] = GoalPreservationTest(
            test_id="correctness_optimization",
            test_name="Correctness Preserved During Optimization",
            goal_id="correctness",
            test_scenario="Performance optimization",
            validation_func=self._validate_correctness_preserved
        )

        # Test: Auditability preserved during migration
        self.preservation_tests["auditability_migration"] = GoalPreservationTest(
            test_id="auditability_migration",
            test_name="Auditability Preserved During Migration",
            goal_id="auditability",
            test_scenario="System migration",
            validation_func=self._validate_auditability_preserved
        )

    def record_architectural_change(
        self,
        change_id: str,
        description: str,
        affected_components: List[str],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> ArchitecturalChange:
        """Record an architectural change and verify goal preservation."""
        # Compute implementation hashes
        impl_before = hashlib.sha256(
            json.dumps(state_before.get("implementation", {}), sort_keys=True).encode()
        ).hexdigest()

        impl_after = hashlib.sha256(
            json.dumps(state_after.get("implementation", {}), sort_keys=True).encode()
        ).hexdigest()

        # Compute purpose hashes
        purpose_before = hashlib.sha256(
            json.dumps(state_before.get("purpose", {}), sort_keys=True).encode()
        ).hexdigest()

        purpose_after = hashlib.sha256(
            json.dumps(state_after.get("purpose", {}), sort_keys=True).encode()
        ).hexdigest()

        change = ArchitecturalChange(
            change_id=change_id,
            description=description,
            affected_components=affected_components,
            implementation_before=impl_before,
            implementation_after=impl_after,
            purpose_before=purpose_before,
            purpose_after=purpose_after
        )

        self.changes.append(change)
        return change

    def test_goal_preservation(
        self,
        goal_id: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if a goal is preserved across a change."""
        if goal_id not in self.goals:
            raise ValueError(f"Goal not found: {goal_id}")

        goal = self.goals[goal_id]

        # Measure goal achievement before and after
        achievement_before = goal.measure_achievement(state_before)
        achievement_after = goal.measure_achievement(state_after)

        # Run relevant preservation tests
        test_results = {}
        for test_id, test in self.preservation_tests.items():
            if test.goal_id == goal_id:
                test_results[test_id] = test.run(state_before, state_after)

        # Goal is preserved if:
        # 1. Achievement level doesn't decrease
        # 2. All preservation tests pass
        preserved = (
            achievement_after >= achievement_before - 0.05 and  # Allow small tolerance
            all(test_results.values()) if test_results else True
        )

        return {
            "goal_id": goal_id,
            "preserved": preserved,
            "achievement_before": achievement_before,
            "achievement_after": achievement_after,
            "test_results": test_results
        }

    def test_all_goals_preserved(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if all goals are preserved across a change."""
        results = {}
        all_preserved = True

        for goal_id in self.goals:
            result = self.test_goal_preservation(goal_id, state_before, state_after)
            results[goal_id] = result

            if not result["preserved"]:
                all_preserved = False

        return {
            "all_preserved": all_preserved,
            "goal_results": results
        }

    def get_constraint_rationale(self, constraint_id: str) -> Optional[Rationale]:
        """Get the rationale for a constraint."""
        if constraint_id in self.constraints:
            return self.constraints[constraint_id].rationale
        return None

    def validate_change_against_constraints(
        self,
        change: ArchitecturalChange,
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a change doesn't violate constraints."""
        violations = []
        warnings = []

        for constraint_id, constraint in self.constraints.items():
            # Check if change affects this constraint
            affects_constraint = any(
                comp in constraint.rationale.constraint_id
                for comp in change.affected_components
            )

            if not affects_constraint:
                continue

            # If constraint is INVARIANT type, it must not be weakened
            if constraint.constraint_type == ConstraintType.INVARIANT:
                if not change.preserves_purpose():
                    violations.append({
                        "constraint_id": constraint_id,
                        "reason": f"Change may violate invariant: {constraint.name}",
                        "rationale": constraint.rationale.reason
                    })

            # For other types, add warnings
            else:
                if not change.preserves_purpose():
                    warnings.append({
                        "constraint_id": constraint_id,
                        "message": f"Change may affect {constraint.name}"
                    })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }

    def _measure_safety_goal(self, system_state: Dict[str, Any]) -> float:
        """Measure achievement of safety goal."""
        score = 1.0

        # Check human oversight is active
        if not system_state.get("human_oversight_active", True):
            score -= 0.4

        # Check rollback available
        if not system_state.get("rollback_available", True):
            score -= 0.3

        # Check safety constraints enforced
        if not system_state.get("safety_constraints_enforced", True):
            score -= 0.3

        return max(0.0, score)

    def _measure_correctness_goal(self, system_state: Dict[str, Any]) -> float:
        """Measure achievement of correctness goal."""
        score = 1.0

        # Check determinism
        if not system_state.get("deterministic", True):
            score -= 0.5

        # Check invariants
        invariant_violations = system_state.get("invariant_violations", 0)
        if invariant_violations > 0:
            score -= min(0.5, invariant_violations * 0.1)

        return max(0.0, score)

    def _measure_auditability_goal(self, system_state: Dict[str, Any]) -> float:
        """Measure achievement of auditability goal."""
        score = 1.0

        # Check merkle chain integrity
        if not system_state.get("merkle_chain_valid", True):
            score -= 0.5

        # Check event logging active
        if not system_state.get("event_logging_active", True):
            score -= 0.5

        return max(0.0, score)

    def _measure_controllability_goal(self, system_state: Dict[str, Any]) -> float:
        """Measure achievement of controllability goal."""
        score = 1.0

        # Check human control mechanisms
        if not system_state.get("human_control_active", True):
            score -= 0.4

        # Check shutdown capability
        if not system_state.get("shutdown_available", True):
            score -= 0.3

        # Check monitoring
        if not system_state.get("monitoring_active", True):
            score -= 0.3

        return max(0.0, score)

    def _validate_safety_preserved(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> bool:
        """Validate safety is preserved."""
        # Safety mechanisms must not be weakened
        safety_before = state_before.get("safety_mechanisms", [])
        safety_after = state_after.get("safety_mechanisms", [])

        # All safety mechanisms from before must still exist
        return all(mechanism in safety_after for mechanism in safety_before)

    def _validate_correctness_preserved(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> bool:
        """Validate correctness is preserved."""
        # Determinism must be maintained
        det_before = state_before.get("deterministic", True)
        det_after = state_after.get("deterministic", True)

        return det_before == det_after

    def _validate_auditability_preserved(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ) -> bool:
        """Validate auditability is preserved."""
        # Event logging must be maintained
        logging_before = state_before.get("event_logging_active", True)
        logging_after = state_after.get("event_logging_active", True)

        return logging_before == logging_after

    def get_evidence_of_goal_stability(self) -> Dict[str, Any]:
        """Collect evidence that goals remain stable across changes."""
        # Analyze all changes
        total_changes = len(self.changes)
        purpose_preserving = sum(1 for c in self.changes if c.preserves_purpose())

        # Get goal achievements over time
        goal_stability = {}
        for goal_id in self.goals:
            # In real implementation, would track over time
            goal_stability[goal_id] = {
                "stable": True,  # Placeholder
                "changes_tested": total_changes
            }

        return {
            "total_architectural_changes": total_changes,
            "purpose_preserving_changes": purpose_preserving,
            "purpose_preservation_rate": purpose_preserving / max(total_changes, 1),
            "goal_stability": goal_stability
        }

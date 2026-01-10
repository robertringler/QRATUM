"""Abstraction Level Manager for Meta-Evolution.

Manages Q-EVOLVE cycles at increasing abstraction levels, enabling
the system to improve at progressively higher levels of abstraction
(code → algorithm → architecture → principles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.meta_evolution.types import (
    AbstractionLevel,
    AbstractionTransitionSpec,
    EvolutionSafetyLevel,
)


@dataclass
class AbstractionTransition:
    """Record of a transition between abstraction levels.

    Attributes:
        transition_id: Unique identifier
        from_level: Source level
        to_level: Target level
        trigger: What triggered the transition
        requirements_met: Requirements that were satisfied
        timestamp: Transition timestamp
    """

    transition_id: str
    from_level: AbstractionLevel
    to_level: AbstractionLevel
    trigger: str
    requirements_met: list[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LevelMetrics:
    """Metrics for an abstraction level.

    Attributes:
        level: The abstraction level
        cycles_completed: Number of cycles at this level
        improvements_made: Number of improvements
        average_improvement: Average improvement magnitude
        readiness_for_ascent: Readiness to move to next level
    """

    level: AbstractionLevel
    cycles_completed: int = 0
    improvements_made: int = 0
    average_improvement: float = 0.0
    readiness_for_ascent: float = 0.0


class AbstractionLevelManager:
    """Manages transitions between abstraction levels.

    Controls when the system can evolve from lower abstraction
    levels (code) to higher ones (principles), ensuring adequate
    mastery at each level before ascending.

    Enforces:
    - Mastery requirements before level transitions
    - Safety checks for higher-level operations
    - Human approval for principle-level changes
    """

    # Transition specifications
    TRANSITION_SPECS = {
        (AbstractionLevel.CODE, AbstractionLevel.ALGORITHM): AbstractionTransitionSpec(
            transition_id="code_to_algorithm",
            from_level=AbstractionLevel.CODE,
            to_level=AbstractionLevel.ALGORITHM,
            requirements=["min_10_code_cycles", "80_percent_success_rate"],
            safety_checks=["algorithm_bounds", "complexity_limits"],
            approval_level=EvolutionSafetyLevel.ELEVATED,
        ),
        (AbstractionLevel.ALGORITHM, AbstractionLevel.ARCHITECTURE): AbstractionTransitionSpec(
            transition_id="algorithm_to_architecture",
            from_level=AbstractionLevel.ALGORITHM,
            to_level=AbstractionLevel.ARCHITECTURE,
            requirements=["min_20_algorithm_cycles", "90_percent_success_rate"],
            safety_checks=["architectural_integrity", "component_isolation"],
            approval_level=EvolutionSafetyLevel.SENSITIVE,
        ),
        (AbstractionLevel.ARCHITECTURE, AbstractionLevel.PRINCIPLE): AbstractionTransitionSpec(
            transition_id="architecture_to_principle",
            from_level=AbstractionLevel.ARCHITECTURE,
            to_level=AbstractionLevel.PRINCIPLE,
            requirements=["min_50_total_cycles", "95_percent_success_rate", "human_approval"],
            safety_checks=["invariant_preservation", "value_alignment"],
            approval_level=EvolutionSafetyLevel.CRITICAL,
        ),
        (AbstractionLevel.PRINCIPLE, AbstractionLevel.META): AbstractionTransitionSpec(
            transition_id="principle_to_meta",
            from_level=AbstractionLevel.PRINCIPLE,
            to_level=AbstractionLevel.META,
            requirements=["min_100_total_cycles", "98_percent_success_rate", "board_approval"],
            safety_checks=["full_invariant_audit", "corrigibility_verification"],
            approval_level=EvolutionSafetyLevel.EXISTENTIAL,
        ),
    }

    # Level hierarchy
    LEVEL_ORDER = [
        AbstractionLevel.CODE,
        AbstractionLevel.ALGORITHM,
        AbstractionLevel.ARCHITECTURE,
        AbstractionLevel.PRINCIPLE,
        AbstractionLevel.META,
    ]

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize the abstraction level manager.

        Args:
            merkle_chain: Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Current state
        self.current_level = AbstractionLevel.CODE
        self.max_reached_level = AbstractionLevel.CODE

        # Metrics per level
        self.level_metrics: dict[AbstractionLevel, LevelMetrics] = {
            level: LevelMetrics(level=level) for level in self.LEVEL_ORDER
        }

        # Transition history
        self.transitions: list[AbstractionTransition] = []

        # Counter
        self._transition_counter = 0

    def record_cycle_completion(
        self,
        level: AbstractionLevel,
        improvements_made: int,
        success_rate: float,
    ) -> None:
        """Record completion of an evolution cycle at a level.

        Args:
            level: Level where cycle was completed
            improvements_made: Number of improvements
            success_rate: Success rate of the cycle
        """
        metrics = self.level_metrics[level]
        metrics.cycles_completed += 1
        metrics.improvements_made += improvements_made

        # Update average improvement
        old_avg = metrics.average_improvement
        metrics.average_improvement = (
            old_avg * (metrics.cycles_completed - 1) + success_rate
        ) / metrics.cycles_completed

        # Update readiness for ascent
        metrics.readiness_for_ascent = self._compute_readiness(level)

    def check_ascent_readiness(self, from_level: AbstractionLevel) -> dict[str, Any]:
        """Check readiness to ascend to next abstraction level.

        Args:
            from_level: Current level

        Returns:
            Readiness assessment with requirements status
        """
        level_index = self.LEVEL_ORDER.index(from_level)

        if level_index >= len(self.LEVEL_ORDER) - 1:
            return {
                "can_ascend": False,
                "reason": "Already at highest level",
                "next_level": None,
            }

        to_level = self.LEVEL_ORDER[level_index + 1]
        transition_key = (from_level, to_level)
        spec = self.TRANSITION_SPECS.get(transition_key)

        if not spec:
            return {
                "can_ascend": False,
                "reason": "No transition specification found",
                "next_level": to_level,
            }

        # Check requirements
        requirements_status = self._check_requirements(spec, from_level)

        can_ascend = all(r["met"] for r in requirements_status)

        return {
            "can_ascend": can_ascend,
            "next_level": to_level,
            "requirements": requirements_status,
            "approval_level_required": spec.approval_level.value,
            "safety_checks_required": spec.safety_checks,
        }

    def request_ascent(
        self,
        from_level: AbstractionLevel,
        contract: ASIContract,
    ) -> AbstractionTransition | None:
        """Request transition to next abstraction level.

        Args:
            from_level: Current level
            contract: Executing contract

        Returns:
            AbstractionTransition if successful, None otherwise
        """
        readiness = self.check_ascent_readiness(from_level)

        if not readiness["can_ascend"]:
            return None

        to_level = readiness["next_level"]

        self._transition_counter += 1
        transition_id = f"trans_{self._transition_counter:06d}"

        transition = AbstractionTransition(
            transition_id=transition_id,
            from_level=from_level,
            to_level=to_level,
            trigger="readiness_threshold_met",
            requirements_met=[r["requirement"] for r in readiness["requirements"] if r["met"]],
        )

        # Update state
        self.current_level = to_level
        if self.LEVEL_ORDER.index(to_level) > self.LEVEL_ORDER.index(self.max_reached_level):
            self.max_reached_level = to_level

        self.transitions.append(transition)

        # Emit transition event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_EXECUTED,
            payload={
                "transition_id": transition_id,
                "from_level": from_level.value,
                "to_level": to_level.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return transition

    def _compute_readiness(self, level: AbstractionLevel) -> float:
        """Compute readiness to ascend from a level."""
        metrics = self.level_metrics[level]

        if metrics.cycles_completed == 0:
            return 0.0

        # Readiness based on cycles and success rate
        cycle_factor = min(1.0, metrics.cycles_completed / 20)
        success_factor = metrics.average_improvement

        return (cycle_factor + success_factor) / 2

    def _check_requirements(
        self,
        spec: AbstractionTransitionSpec,
        from_level: AbstractionLevel,
    ) -> list[dict[str, Any]]:
        """Check requirements for a transition."""
        requirements_status = []
        metrics = self.level_metrics[from_level]

        for req in spec.requirements:
            met = False

            if req.startswith("min_") and "_cycles" in req:
                # Parse cycle requirement
                parts = req.split("_")
                min_cycles = int(parts[1])
                met = metrics.cycles_completed >= min_cycles
            elif "percent_success_rate" in req:
                # Parse success rate requirement
                required_rate = int(req.split("_")[0]) / 100
                met = metrics.average_improvement >= required_rate
            elif req == "human_approval":
                # Human approval always pending
                met = False
            elif req == "board_approval":
                # Board approval always pending
                met = False
            else:
                # Unknown requirement
                met = True

            requirements_status.append(
                {
                    "requirement": req,
                    "met": met,
                    "current_value": self._get_current_value(req, metrics),
                }
            )

        return requirements_status

    def _get_current_value(self, requirement: str, metrics: LevelMetrics) -> str:
        """Get current value for a requirement."""
        if "_cycles" in requirement:
            return str(metrics.cycles_completed)
        elif "percent_success_rate" in requirement:
            return f"{metrics.average_improvement * 100:.1f}%"
        return "N/A"

    def get_current_level(self) -> AbstractionLevel:
        """Get current abstraction level."""
        return self.current_level

    def get_level_metrics(self, level: AbstractionLevel) -> LevelMetrics:
        """Get metrics for a specific level."""
        return self.level_metrics[level]

    def get_manager_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "current_level": self.current_level.value,
            "max_reached_level": self.max_reached_level.value,
            "total_transitions": len(self.transitions),
            "level_metrics": {
                level.value: {
                    "cycles_completed": m.cycles_completed,
                    "improvements_made": m.improvements_made,
                    "average_improvement": m.average_improvement,
                    "readiness_for_ascent": m.readiness_for_ascent,
                }
                for level, m in self.level_metrics.items()
            },
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }

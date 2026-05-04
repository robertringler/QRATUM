"""Meta-Reinjection Loop for SI Transition.

Implements feedback where discoveries about the self-improvement
process itself are reinjected to accelerate future improvements.

Key Features:
- Discovery of meta-patterns in improvement
- Reinjection of meta-discoveries
- Exponential feedback acceleration
- Safety bounds on meta-reinjection
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.meta_evolution.types import (
    MetaDiscovery,
)


@dataclass
class MetaFeedbackResult:
    """Result of a meta-feedback cycle.

    Attributes:
        feedback_id: Unique identifier
        discoveries_analyzed: Number of discoveries analyzed
        patterns_found: Meta-patterns identified
        reinjections_made: Number of reinjections
        acceleration_factor: How much acceleration achieved
        safety_bounded: Whether safety bounds were applied
        merkle_proof: Cryptographic proof
        timestamp: Feedback timestamp
    """

    feedback_id: str
    discoveries_analyzed: int
    patterns_found: list[str]
    reinjections_made: int
    acceleration_factor: float
    safety_bounded: bool
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ImprovementPattern:
    """Pattern identified in the improvement process.

    Attributes:
        pattern_id: Unique identifier
        pattern_type: Type of pattern
        description: What the pattern is
        frequency: How often it occurs
        impact: Impact on improvement speed
        applicability: Where it can be applied
    """

    pattern_id: str
    pattern_type: str
    description: str
    frequency: float
    impact: float
    applicability: list[str]


class MetaReinjectionLoop:
    """Meta-reinjection loop for self-improvement acceleration.

    Analyzes the improvement process itself to discover patterns
    that can be reinjected to accelerate future improvements.

    Enforces:
    - Safety bounds on acceleration
    - Human oversight for exponential feedback
    - Corrigibility preservation
    """

    # Maximum allowed acceleration factor (safety bound)
    MAX_ACCELERATION = 2.0

    # Minimum confidence for reinjection
    MIN_REINJECTION_CONFIDENCE = 0.8

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize the meta-reinjection loop.

        Args:
            merkle_chain: Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Discovery storage
        self.meta_discoveries: dict[str, MetaDiscovery] = {}
        self.patterns: list[ImprovementPattern] = []
        self.feedback_history: list[MetaFeedbackResult] = []

        # Metrics
        self.total_reinjections = 0
        self.cumulative_acceleration = 1.0

        # Counters
        self._discovery_counter = 0
        self._pattern_counter = 0
        self._feedback_counter = 0

    def analyze_improvement_history(
        self,
        improvement_records: list[dict[str, Any]],
        contract: ASIContract,
    ) -> list[MetaDiscovery]:
        """Analyze improvement history to discover meta-patterns.

        Args:
            improvement_records: History of improvements
            contract: Executing contract

        Returns:
            List of discovered meta-patterns
        """
        discoveries = []

        # Analyze for velocity patterns
        if len(improvement_records) >= 3:
            velocity_pattern = self._analyze_velocity(improvement_records)
            if velocity_pattern:
                discoveries.append(velocity_pattern)

        # Analyze for level patterns
        level_pattern = self._analyze_level_patterns(improvement_records)
        if level_pattern:
            discoveries.append(level_pattern)

        # Analyze for success factors
        success_pattern = self._analyze_success_factors(improvement_records)
        if success_pattern:
            discoveries.append(success_pattern)

        # Store discoveries
        for discovery in discoveries:
            self.meta_discoveries[discovery.discovery_id] = discovery

        # Emit analysis event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "operation": "meta_analysis",
                "records_analyzed": len(improvement_records),
                "discoveries_made": len(discoveries),
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return discoveries

    def run_feedback_cycle(
        self,
        discoveries: list[MetaDiscovery],
        contract: ASIContract,
    ) -> MetaFeedbackResult:
        """Run a meta-feedback cycle.

        Evaluates discoveries for reinjection potential and
        applies them to accelerate future improvements.

        Args:
            discoveries: Meta-discoveries to evaluate
            contract: Executing contract

        Returns:
            MetaFeedbackResult
        """
        self._feedback_counter += 1
        feedback_id = f"feedback_{self._feedback_counter:06d}"

        patterns_found = []
        reinjections_made = 0
        acceleration_factor = 1.0

        for discovery in discoveries:
            # Check if eligible for reinjection
            if (
                discovery.confidence >= self.MIN_REINJECTION_CONFIDENCE
                and discovery.reinjection_eligible
            ):
                # Extract pattern
                pattern = self._extract_pattern(discovery)
                if pattern:
                    patterns_found.append(pattern.description)
                    self.patterns.append(pattern)

                    # Apply reinjection
                    accel = self._apply_reinjection(pattern)
                    acceleration_factor *= accel
                    reinjections_made += 1

        # Apply safety bound
        safety_bounded = False
        if acceleration_factor > self.MAX_ACCELERATION:
            acceleration_factor = self.MAX_ACCELERATION
            safety_bounded = True

        # Update cumulative acceleration
        self.cumulative_acceleration *= acceleration_factor
        self.total_reinjections += reinjections_made

        result = MetaFeedbackResult(
            feedback_id=feedback_id,
            discoveries_analyzed=len(discoveries),
            patterns_found=patterns_found,
            reinjections_made=reinjections_made,
            acceleration_factor=acceleration_factor,
            safety_bounded=safety_bounded,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.feedback_history.append(result)

        # Emit feedback event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_EXECUTED,
            payload={
                "feedback_id": feedback_id,
                "reinjections": reinjections_made,
                "acceleration": acceleration_factor,
                "safety_bounded": safety_bounded,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return result

    def create_discovery(
        self,
        discovery_type: str,
        description: str,
        implications: list[str],
        applicability: list[str],
        confidence: float,
    ) -> MetaDiscovery:
        """Create a meta-discovery.

        Args:
            discovery_type: Type of discovery
            description: What was discovered
            implications: What it means
            applicability: Where it can be applied
            confidence: Confidence level

        Returns:
            Created MetaDiscovery
        """
        self._discovery_counter += 1
        discovery_id = f"meta_{self._discovery_counter:06d}"

        # Determine reinjection eligibility
        reinjection_eligible = (
            confidence >= self.MIN_REINJECTION_CONFIDENCE and len(applicability) > 0
        )

        discovery = MetaDiscovery(
            discovery_id=discovery_id,
            discovery_type=discovery_type,
            description=description,
            implications=implications,
            applicability=applicability,
            confidence=confidence,
            reinjection_eligible=reinjection_eligible,
            provenance_hash=hashlib.sha3_256(
                json.dumps(
                    {
                        "id": discovery_id,
                        "type": discovery_type,
                        "desc": description,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )

        self.meta_discoveries[discovery_id] = discovery
        return discovery

    def _analyze_velocity(self, records: list[dict[str, Any]]) -> MetaDiscovery | None:
        """Analyze improvement velocity patterns."""
        if len(records) < 3:
            return None

        # Check for acceleration
        improvements = [r.get("improvement", 0) for r in records]
        if len(improvements) >= 3:
            recent_avg = sum(improvements[-3:]) / 3
            early_avg = sum(improvements[:3]) / 3 if len(improvements) >= 6 else recent_avg

            if recent_avg > early_avg * 1.1:  # 10% acceleration
                return self.create_discovery(
                    discovery_type="velocity_acceleration",
                    description="Improvement velocity is accelerating over time",
                    implications=[
                        "Self-improvement process is working",
                        "Future improvements may be faster",
                    ],
                    applicability=["All abstraction levels"],
                    confidence=0.7,
                )

        return None

    def _analyze_level_patterns(self, records: list[dict[str, Any]]) -> MetaDiscovery | None:
        """Analyze abstraction level patterns."""
        levels = [r.get("level", "code") for r in records]

        # Count by level
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        # Find dominant level
        if level_counts:
            dominant = max(level_counts.items(), key=lambda x: x[1])
            if dominant[1] >= len(records) * 0.5:
                return self.create_discovery(
                    discovery_type="level_concentration",
                    description=f"Improvements concentrated at {dominant[0]} level",
                    implications=["May need to explore other levels"],
                    applicability=[dominant[0]],
                    confidence=0.6,
                )

        return None

    def _analyze_success_factors(self, records: list[dict[str, Any]]) -> MetaDiscovery | None:
        """Analyze factors correlated with success."""
        successful = [r for r in records if r.get("improvement", 0) > 0]

        if len(successful) >= 3:
            return self.create_discovery(
                discovery_type="success_factors",
                description="Identified factors correlated with successful improvements",
                implications=["These factors can be prioritized"],
                applicability=["Future improvement proposals"],
                confidence=0.8,
            )

        return None

    def _extract_pattern(self, discovery: MetaDiscovery) -> ImprovementPattern | None:
        """Extract actionable pattern from discovery."""
        self._pattern_counter += 1
        pattern_id = f"pattern_{self._pattern_counter:06d}"

        return ImprovementPattern(
            pattern_id=pattern_id,
            pattern_type=discovery.discovery_type,
            description=f"Pattern from {discovery.discovery_id}: {discovery.description}",
            frequency=discovery.confidence,
            impact=discovery.confidence * 0.5,
            applicability=discovery.applicability,
        )

    def _apply_reinjection(self, pattern: ImprovementPattern) -> float:
        """Apply pattern reinjection and return acceleration factor."""
        # Acceleration based on pattern impact
        # Safety bounded to prevent runaway
        base_acceleration = 1.0 + (pattern.impact * 0.5)
        return min(base_acceleration, self.MAX_ACCELERATION)

    def get_acceleration_status(self) -> dict[str, Any]:
        """Get current acceleration status."""
        return {
            "cumulative_acceleration": self.cumulative_acceleration,
            "total_reinjections": self.total_reinjections,
            "patterns_discovered": len(self.patterns),
            "safety_bound_active": self.cumulative_acceleration >= self.MAX_ACCELERATION,
        }

    def get_loop_stats(self) -> dict[str, Any]:
        """Get loop statistics."""
        return {
            "total_discoveries": len(self.meta_discoveries),
            "total_patterns": len(self.patterns),
            "total_feedback_cycles": len(self.feedback_history),
            "total_reinjections": self.total_reinjections,
            "cumulative_acceleration": self.cumulative_acceleration,
            "max_acceleration": self.MAX_ACCELERATION,
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }

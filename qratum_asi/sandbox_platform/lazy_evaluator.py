"""Lazy Evaluator for Critical Subsystem Proposals.

Implements lazy evaluation to only run critical subsystem proposals,
reducing unnecessary computation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain

from qratum_asi.sandbox_platform.types import SandboxProposal, ProposalPriority


class CriticalityLevel(Enum):
    """Level of criticality for evaluation."""

    NONE = "none"  # Not critical, skip
    LOW = "low"  # Low criticality, defer
    MEDIUM = "medium"  # Medium criticality, evaluate when resources allow
    HIGH = "high"  # High criticality, evaluate soon
    CRITICAL = "critical"  # Critical, evaluate immediately


class EvaluationPolicy(Enum):
    """Policy for lazy evaluation."""

    ALWAYS = "always"  # Always evaluate
    CRITICAL_ONLY = "critical_only"  # Only evaluate critical
    ON_DEMAND = "on_demand"  # Evaluate when explicitly requested
    THRESHOLD = "threshold"  # Evaluate if above threshold


@dataclass
class CriticalityAssessment:
    """Assessment of proposal criticality.

    Attributes:
        assessment_id: Unique assessment identifier
        proposal_id: Assessed proposal ID
        criticality: Assessed criticality level
        score: Criticality score (0-1)
        factors: Factors contributing to assessment
        should_evaluate: Whether proposal should be evaluated
        defer_until: If deferred, when to re-evaluate
    """

    assessment_id: str
    proposal_id: str
    criticality: CriticalityLevel
    score: float
    factors: dict[str, float]
    should_evaluate: bool
    defer_until: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize assessment."""
        return {
            "assessment_id": self.assessment_id,
            "proposal_id": self.proposal_id,
            "criticality": self.criticality.value,
            "score": self.score,
            "factors": self.factors,
            "should_evaluate": self.should_evaluate,
            "defer_until": self.defer_until,
            "timestamp": self.timestamp,
        }


@dataclass
class LazyEvaluationResult:
    """Result of lazy evaluation decision.

    Attributes:
        result_id: Unique result identifier
        proposal_id: Proposal ID
        evaluated: Whether evaluation was performed
        result: Evaluation result if evaluated
        skipped_reason: Reason if skipped
        saved_time_ms: Estimated time saved by skipping
    """

    result_id: str
    proposal_id: str
    evaluated: bool
    result: dict[str, Any] | None = None
    skipped_reason: str = ""
    saved_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize lazy evaluation result."""
        return {
            "result_id": self.result_id,
            "proposal_id": self.proposal_id,
            "evaluated": self.evaluated,
            "has_result": self.result is not None,
            "skipped_reason": self.skipped_reason,
            "saved_time_ms": self.saved_time_ms,
            "timestamp": self.timestamp,
        }


class LazyEvaluator:
    """Lazy evaluator for sandbox proposals.

    Only evaluates critical subsystem proposals, skipping or
    deferring non-critical evaluations to save resources.
    """

    def __init__(
        self,
        evaluator_id: str = "lazy",
        policy: EvaluationPolicy = EvaluationPolicy.THRESHOLD,
        criticality_threshold: float = 0.5,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize lazy evaluator.

        Args:
            evaluator_id: Unique evaluator identifier
            policy: Evaluation policy
            criticality_threshold: Threshold for THRESHOLD policy
            merkle_chain: Merkle chain for audit trail
        """
        self.evaluator_id = evaluator_id
        self.policy = policy
        self.criticality_threshold = criticality_threshold
        self.merkle_chain = merkle_chain or MerkleChain()

        # Critical subsystems
        self._critical_subsystems: set[str] = {
            "core_engine",
            "security",
            "safety",
            "governance",
            "audit",
        }

        # Sensitive data keys that indicate critical evaluation needed
        self._sensitive_data_keys: set[str] = {
            "credentials",
            "secrets",
            "patient_data",
            "phi",
            "pii",
            "ssn",
            "api_key",
        }

        # Assessment cache
        self._assessments: dict[str, CriticalityAssessment] = {}
        self._assessment_counter = 0
        self._result_counter = 0

        # Deferred evaluations
        self._deferred: list[tuple[str, str]] = []  # (proposal_id, defer_until)

        # Custom assessment function
        self._custom_assessor: Callable[[SandboxProposal], CriticalityAssessment] | None = None

        # Statistics
        self._total_proposals = 0
        self._evaluated_proposals = 0
        self._skipped_proposals = 0
        self._deferred_proposals = 0
        self._total_time_saved_ms = 0.0

        # Log initialization
        self.merkle_chain.add_event(
            "lazy_evaluator_initialized",
            {
                "evaluator_id": evaluator_id,
                "policy": policy.value,
                "criticality_threshold": criticality_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_critical_subsystems(self, subsystems: set[str]) -> None:
        """Set the list of critical subsystems.

        Args:
            subsystems: Set of critical subsystem names
        """
        self._critical_subsystems = subsystems

    def add_critical_subsystem(self, subsystem: str) -> None:
        """Add a critical subsystem.

        Args:
            subsystem: Subsystem name to add
        """
        self._critical_subsystems.add(subsystem)

    def set_sensitive_data_keys(self, keys: set[str]) -> None:
        """Set the list of sensitive data keys.

        Args:
            keys: Set of key names that indicate sensitive data
        """
        self._sensitive_data_keys = keys

    def add_sensitive_data_key(self, key: str) -> None:
        """Add a sensitive data key.

        Args:
            key: Key name to add
        """
        self._sensitive_data_keys.add(key)

    def set_custom_assessor(
        self,
        assessor: Callable[[SandboxProposal], CriticalityAssessment],
    ) -> None:
        """Set custom criticality assessor.

        Args:
            assessor: Custom assessment function
        """
        self._custom_assessor = assessor

    def assess_criticality(self, proposal: SandboxProposal) -> CriticalityAssessment:
        """Assess criticality of a proposal.

        Args:
            proposal: Proposal to assess

        Returns:
            CriticalityAssessment for the proposal
        """
        self._assessment_counter += 1
        assessment_id = f"assess_{self.evaluator_id}_{self._assessment_counter:08d}"

        if self._custom_assessor:
            assessment = self._custom_assessor(proposal)
            assessment.assessment_id = assessment_id
            return assessment

        # Default assessment logic
        factors: dict[str, float] = {}

        # Factor: Target subsystems
        targets_critical = any(
            t in self._critical_subsystems for t in proposal.target_subsystems
        )
        factors["critical_target"] = 1.0 if targets_critical else 0.0

        # Factor: Priority
        priority_scores = {
            ProposalPriority.CRITICAL: 1.0,
            ProposalPriority.HIGH: 0.8,
            ProposalPriority.NORMAL: 0.5,
            ProposalPriority.LOW: 0.2,
        }
        factors["priority"] = priority_scores.get(proposal.priority, 0.5)

        # Factor: Estimated impact
        factors["impact"] = proposal.estimated_impact

        # Factor: Has sensitive data
        has_sensitive = any(
            k in proposal.payload
            for k in self._sensitive_data_keys
        )
        factors["sensitive"] = 1.0 if has_sensitive else 0.0

        # Compute overall score
        weights = {
            "critical_target": 0.35,
            "priority": 0.25,
            "impact": 0.25,
            "sensitive": 0.15,
        }
        score = sum(factors[k] * weights[k] for k in factors)

        # Determine criticality level
        if score >= 0.8:
            criticality = CriticalityLevel.CRITICAL
        elif score >= 0.6:
            criticality = CriticalityLevel.HIGH
        elif score >= 0.4:
            criticality = CriticalityLevel.MEDIUM
        elif score >= 0.2:
            criticality = CriticalityLevel.LOW
        else:
            criticality = CriticalityLevel.NONE

        # Determine if should evaluate
        should_evaluate = self._should_evaluate(criticality, score)

        assessment = CriticalityAssessment(
            assessment_id=assessment_id,
            proposal_id=proposal.proposal_id,
            criticality=criticality,
            score=score,
            factors=factors,
            should_evaluate=should_evaluate,
        )

        self._assessments[proposal.proposal_id] = assessment
        return assessment

    def _should_evaluate(
        self,
        criticality: CriticalityLevel,
        score: float,
    ) -> bool:
        """Determine if proposal should be evaluated based on policy."""
        if self.policy == EvaluationPolicy.ALWAYS:
            return True

        if self.policy == EvaluationPolicy.CRITICAL_ONLY:
            return criticality in (CriticalityLevel.CRITICAL, CriticalityLevel.HIGH)

        if self.policy == EvaluationPolicy.ON_DEMAND:
            return False  # Explicit request required

        if self.policy == EvaluationPolicy.THRESHOLD:
            return score >= self.criticality_threshold

        return True

    def evaluate(
        self,
        proposal: SandboxProposal,
        evaluator: Callable[[SandboxProposal], dict[str, Any]],
        force: bool = False,
        estimated_time_ms: float = 100.0,
    ) -> LazyEvaluationResult:
        """Lazily evaluate a proposal.

        Args:
            proposal: Proposal to evaluate
            evaluator: Evaluation function
            force: Force evaluation regardless of criticality
            estimated_time_ms: Estimated evaluation time

        Returns:
            LazyEvaluationResult
        """
        self._total_proposals += 1
        self._result_counter += 1
        result_id = f"lazy_{self.evaluator_id}_{self._result_counter:08d}"

        # Assess criticality
        assessment = self.assess_criticality(proposal)

        # Determine if should evaluate
        if not force and not assessment.should_evaluate:
            self._skipped_proposals += 1
            self._total_time_saved_ms += estimated_time_ms

            # Log skip
            self.merkle_chain.add_event(
                "lazy_evaluation_skipped",
                {
                    "proposal_id": proposal.proposal_id,
                    "criticality": assessment.criticality.value,
                    "score": assessment.score,
                },
            )

            return LazyEvaluationResult(
                result_id=result_id,
                proposal_id=proposal.proposal_id,
                evaluated=False,
                skipped_reason=f"Below criticality threshold ({assessment.score:.2f} < {self.criticality_threshold})",
                saved_time_ms=estimated_time_ms,
            )

        # Perform evaluation
        start_time = time.perf_counter()
        try:
            result = evaluator(proposal)
            self._evaluated_proposals += 1

            # Log evaluation
            self.merkle_chain.add_event(
                "lazy_evaluation_performed",
                {
                    "proposal_id": proposal.proposal_id,
                    "criticality": assessment.criticality.value,
                },
            )

            return LazyEvaluationResult(
                result_id=result_id,
                proposal_id=proposal.proposal_id,
                evaluated=True,
                result=result,
            )

        except Exception as e:
            return LazyEvaluationResult(
                result_id=result_id,
                proposal_id=proposal.proposal_id,
                evaluated=True,
                result={"error": str(e)},
            )

    def defer(
        self,
        proposal: SandboxProposal,
        defer_until: str,
    ) -> None:
        """Defer evaluation of a proposal.

        Args:
            proposal: Proposal to defer
            defer_until: ISO timestamp to re-evaluate
        """
        self._deferred.append((proposal.proposal_id, defer_until))
        self._deferred_proposals += 1

        # Update assessment
        if proposal.proposal_id in self._assessments:
            self._assessments[proposal.proposal_id].defer_until = defer_until

    def get_deferred(self) -> list[tuple[str, str]]:
        """Get list of deferred evaluations."""
        return self._deferred.copy()

    def get_assessment(self, proposal_id: str) -> CriticalityAssessment | None:
        """Get assessment for a proposal."""
        return self._assessments.get(proposal_id)

    def get_evaluator_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        skip_rate = (
            self._skipped_proposals / self._total_proposals
            if self._total_proposals > 0
            else 0
        )

        return {
            "evaluator_id": self.evaluator_id,
            "policy": self.policy.value,
            "criticality_threshold": self.criticality_threshold,
            "critical_subsystems": list(self._critical_subsystems),
            "total_proposals": self._total_proposals,
            "evaluated_proposals": self._evaluated_proposals,
            "skipped_proposals": self._skipped_proposals,
            "deferred_proposals": self._deferred_proposals,
            "skip_rate": skip_rate,
            "total_time_saved_ms": self._total_time_saved_ms,
            "assessment_count": len(self._assessments),
        }
